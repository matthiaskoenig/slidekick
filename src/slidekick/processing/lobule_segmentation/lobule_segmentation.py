import numpy as np
from pathlib import Path
from rich.prompt import Confirm
from typing import List, Union, Optional, Tuple, Iterable, Dict
from PIL import Image
import cv2
import datetime
import uuid
from skimage.filters import threshold_multiotsu
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH

from slidekick.processing.roi.roi_utils import largest_bbox, ensure_grayscale_uint8, crop_image, detect_tissue_mask

from slidekick.processing.lobule_segmentation.run_skeletize_image import run_skeletize_image
from slidekick.processing.lobule_segmentation.get_segments import segment_thinned_image
from process_segments import process_segments_to_mask

def multiotsu_split(gray: np.ndarray, classes: int = 3, blur_sigma: float = 1.5):
    """
    3-class Otsu on a blurred grayscale -> labels in {0,1,2}.
    Heuristic: lowest mean = true background, middle = microscope bg, highest = tissue.
    """
    g = gray.astype(np.uint8)
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    # thresholds length = classes-1 -> two thresholds for 3 classes
    thr = threshold_multiotsu(g, classes=classes)
    lbl = np.digitize(gray, bins=thr)  # uses original gray for sharper boundaries
    # order by class mean
    means = [gray[lbl == i].mean() if np.any(lbl == i) else -1 for i in range(classes)]
    order = np.argsort(means)  # low -> high
    idx_true_bg, idx_mic_bg, idx_tissue = order[0], order[1], order[-1]
    return lbl.astype(np.int32), (idx_true_bg, idx_mic_bg, idx_tissue)


def detect_tissue_mask_multiotsu(gray: np.ndarray,
                                 morphological_radius: int = 5,
                                 min_size: int = 10_000,
                                 hole_size: int = 10_000):
    """
    Build boolean masks: tissue, microscope background, true background.
    Cleans with closing + small object/hole removal.
    """
    lbl, (i0, i1, i2) = multiotsu_split(gray, classes=3, blur_sigma=1.5)
    m_tis  = (lbl == i2)

    # morph clean tissue
    m_tis = closing(m_tis, disk(int(morphological_radius)))
    m_tis = remove_small_objects(m_tis, min_size=min_size)
    m_tis = remove_small_holes(m_tis, area_threshold=hole_size)

    return m_tis.astype(bool)


def _overlay_skeleton(image_stack: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """Red overlay of skeleton on mean of channels."""
    base = image_stack.mean(axis=2).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    rgb[skeleton > 0] = (255, 0, 0)
    return rgb


def _filter(image_stack: np.ndarray) -> np.ndarray:
    """
    Channel-wise version of zia's filtering. Works for N >= 1 channels.
    Keeps dtype uint8, returns (H, W, N).
    """

    if image_stack.ndim != 3:
        raise ValueError(f"Expected (H, W, N) stack, got {image_stack.shape}")

    H, W, N = image_stack.shape

    # 1) set missing to 0 (any zero across channels -> zero all channels)
    missing_mask = np.any(image_stack == 0, axis=-1)

    out = np.empty((H, W, N), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    for c in range(N):
        ch = image_stack[..., c]

        # ensure uint8 for OpenCV ops
        if ch.dtype != np.uint8:
            ch = np.clip(ch, 0, 255).astype(np.uint8)

        # apply missing mask on this channel
        ch[missing_mask] = 0

        # 2) convolution
        ch = cv2.medianBlur(ch, ksize=7)

        # 3) adaptive histogram norm (2D only)
        ch = clahe.apply(ch)
        ch[ch < 10] = 0  # suppress background altered by CLAHE

        # 4) convolution
        ch = cv2.medianBlur(ch, ksize=3)

        # 5) channel normalization to 99th percentile
        # use >0 to avoid zero-inflation from background
        nz = ch[ch > 0]
        p99 = float(np.percentile(nz, 99)) if nz.size else 255.0
        if p99 <= 0:
            p99 = 1.0
        ch = (ch.astype(np.float32) / p99 * 255.0).clip(0, 255).astype(np.uint8)

        out[..., c] = ch

    return out


def build_mask_pyramid_from_processed(
    mask_cropped: np.ndarray,
    img_size_base: Tuple[int, int],            # (Hb, Wb) cropped ROI at base_level (AFTER bbox crop, BEFORE padding)
    bbox_base: Tuple[int, int, int, int],      # (min_r, min_c, max_r, max_c) in base_level coords
    orig_shapes: Dict[int, Tuple[int, int]],   # {level: (H,W)} full-frame shapes at each level
    base_level: int,                           # the level used to load/crop
) -> Dict[int, np.ndarray]:
    """
    0) mask_cropped: mask with padding already stripped (so shape == processed ROI size without pad)
    1) Resize mask_cropped from processed-ROI size -> base-level ROI size (img_size_base)
    2) Paste into a full-size base_level canvas at bbox_base
    3) Resample that base canvas to every level in orig_shapes (NEAREST to preserve labels)
    Returns {level: full_mask_at_level}
    """
    # Step 1: processed ROI -> base-level ROI
    Hb, Wb = int(img_size_base[0]), int(img_size_base[1])      # target ROI size at base_level
    if Hb <= 0 or Wb <= 0 or mask_cropped.size == 0:
        return {lvl: np.zeros(orig_shapes[lvl], dtype=np.int32) for lvl in orig_shapes}

    roi_base = cv2.resize(
        mask_cropped.astype(np.int32),
        (Wb, Hb),  # (width, height)
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    # Step 2: paste into full-size base canvas at bbox
    Hfull_base, Wfull_base = orig_shapes[base_level]
    min_r, min_c, max_r, max_c = bbox_base
    canvas_base = np.zeros((Hfull_base, Wfull_base), dtype=np.int32)
    # Safety clamp (in case of off-by-one)
    min_r = max(0, min_r); min_c = max(0, min_c)
    max_r = min(Hfull_base, max_r); max_c = min(Wfull_base, max_c)
    if (max_r - min_r) != Hb or (max_c - min_c) != Wb:
        # If bbox dims and img_size_base mismatch by 1 px, reconcile by resize
        Hb2, Wb2 = (max_r - min_r), (max_c - min_c)
        roi_base = cv2.resize(roi_base, (Wb2, Hb2), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    canvas_base[min_r:max_r, min_c:max_c] = roi_base

    # -- Step 3: build pyramid by resizing base canvas to each level
    out = {}
    for lvl, (Hdst, Wdst) in orig_shapes.items():
        if lvl == base_level:
            out[lvl] = canvas_base.copy()
        else:
            out[lvl] = cv2.resize(
                canvas_base, (Wdst, Hdst), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
    return out


class LobuleSegmentor(BaseOperator):
    """
    Based on https://github.com/matthiaskoenig/zonation-image-analysis/blob/develop/src/zia/pipeline/pipeline_components/segementation_component.py
    """

    def __init__(self,
                 metadata: Union[Metadata, List[Metadata]],
                 channel_selection: Union[int, List[int]] = None,
                 throw_out_ratio: float = None,
                 preview: bool = True,
                 confirm: bool = True,
                 base_level: int = 0,
                 region_size: int = 20,
                 target_superpixels: int = None):
        """
        @param metadata: List or single metadata object to load and use for lobule segmentation
        @param channel_selection: Number of Metadata objects that should be inverted. If None, invert none.
        Images should be inverted so that high absorbance, i.e., bright values are pericentral zones.
        Different to zia where simply high absorbance = bright
        @param throw_out_ratio: the min percentage of non_background pixel for the slide to have to not be discarded
        @param preview: whether to preview the segmentation
        @param confirm: whether to confirm the segmentation
        @param base_level: Pyramid level to load
        @param region_size: average size of superpixels for SLIC in pixels
        @param target_superpixels: number of superpixels to use for segmentation, overrides region_size
        """
        self.throw_out_ratio = throw_out_ratio
        self.preview = preview
        self.confirm = confirm
        self.base_level = base_level
        self.region_size = region_size
        self.target_superpixels = target_superpixels
        # make sure channel is list for later iteration
        if isinstance(channel_selection, int):
            channel_selection = [channel_selection]
        elif channel_selection is None:
            channel_selection = []
        super().__init__(metadata, channel_selection=None)
        # Override init back to have direct info on channels to invert
        self.channels = channel_selection

    @staticmethod
    def _downsample_to_max_side(img: np.ndarray, max_side: int = 2048) -> np.ndarray:
        """
        Downsample image so max(height, width) == max_side, preserving aspect ratio.
        Uses Pillow LANCZOS for high-quality downscale.
        No-op if the image is already smaller than max_side.
        """

        # Ensure we operate on HxW or HxWxC ndarray
        if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
            raise ValueError("Preview expects an ndarray image of shape HxW or HxWxC.")

        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return img  # already small enough

        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Convert to PIL, resize, back to numpy
        # Normalize dtype to uint8 for display if needed
        arr = img
        if arr.dtype != np.uint8:
            # clip to [0,255] then cast for stable visualization
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        pil = Image.fromarray(arr)
        pil = pil.resize((new_w, new_h), resample=Image.LANCZOS)
        out = np.asarray(pil)

        return out

    def _preview_images(self, images: List[np.ndarray], titles: Optional[List[str]] = None) -> None:
        """
        Downsample and render a grid preview. Max 2048 px on the longest side per image.
        """
        if not images:
            console.print("No images to preview.", style="warning")
            return

        thumbs = [self._downsample_to_max_side(im, 2048) for im in images]

        # Basic layout: up to 3 columns. Compute rows to fit all images.
        n = len(thumbs)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

        # Reasonable figure size based on columns/rows
        fig_w = 5 * cols
        fig_h = 5 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)

        for idx, ax in enumerate(axes.flat):
            ax.axis("off")
            if idx >= n:
                continue
            im = thumbs[idx]
            # grayscale vs RGB handling
            if im.ndim == 2:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im)
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=9)

        plt.show()

    def _load_and_invert_images_from_metadatas(self) -> np.ndarray:
        """
        Load each image at the pyramid level defined, invert selected channels,
        harmonize shapes, and stack along the last axis -> (H, W, N).
        """

        arrays = []

        for i, md in enumerate(self.metadata):
            try:
                img = self.load_image(i)[self.base_level]  # Load resolution
            except:
                raise Exception(f"Level {self.base_level} failed for image {i} in stack")
            # ensure 2D
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = np.mean(img[..., :3], axis=-1).astype(img.dtype)

            arrays.append(img)

        if not arrays:
            raise RuntimeError("No images loaded after filtering.")

        # stack along the third dimension
        # harmonize shapes before stacking
        h_min = min(a.shape[0] for a in arrays)
        w_min = min(a.shape[1] for a in arrays)
        arrays = [a[:h_min, :w_min] for a in arrays]

        stack = np.dstack(arrays).astype(np.uint8)

        # Get resolutions / image sizes as dict for later back mapping based on metadata object 0
        orig_shapes = {k: v.shape[:2] for k, v in self.load_image(0).items()}

        # ROI detection block
        console.print("Complete. Detecting ROI...", style="info")
        stack_grayscale = ensure_grayscale_uint8(stack)
        # We detect three levels in each wsi: actual tissue, black background and the background in microscopy.
        # returns (tissue_mask, microscope_bg_mask, true_bg_mask)
        tissue_mask = detect_tissue_mask_multiotsu(
            stack_grayscale, morphological_radius=5, min_size=10_000, hole_size=10_000
        )

        # hard clamp everything outside tissue to black
        stack[~tissue_mask] = 0

        # crop to tissue bbox
        bbox = largest_bbox(tissue_mask.astype(np.uint8))
        stack = crop_image(stack, bbox)

        # Inversion and discard only AFTER cropping:
        for i in range(stack.shape[-1]):

            # optional discard by foreground coverage
            if self.throw_out_ratio is not None:
                non_bg = np.count_nonzero(img != 255)
                ratio = non_bg / stack[:, :, i].size
                if ratio < self.throw_out_ratio:
                    console.print(
                        f"Discarded {i} with non-background pixel ratio: {ratio:.3f}",
                        style="warning",
                    )
                    stack = np.delete(stack, i, axis=-1)

            # optional inversion
            if self.channels is not None and i in self.channels:
                if np.issubdtype(stack[:, :, i].dtype, np.integer):
                    info = np.iinfo(stack[:, :, i].dtype)
                    stack[:, :, i] = info.max - stack[:, :, i]
                else:  # float images assumed in [0,1]
                    stack[:, :, i] = 1.0 - stack[:, :, i]

        return stack, bbox, orig_shapes

    def apply(self) -> Metadata:

        console.print("Loading images...", style="info")
        # Load images and invert based on metadata
        img_stack, bbox, orig_shapes = self._load_and_invert_images_from_metadatas()

        img_size_base = img_stack.shape[:2]  # base size for back-mapping of mask

        console.print("Complete. Now applying filters...", style="info")

        # Apply filters
        img_stack = _filter(img_stack)

        # Show Preview after loading and filtering
        if self.preview:
            # Ensure we have a list of arrays to preview
            # numpy array -> split into list
            if isinstance(img_stack, list):
                imgs = img_stack
            elif hasattr(img_stack, "shape"):
                if img_stack.ndim == 2:
                    imgs = [np.asarray(img_stack)]
                elif img_stack.ndim == 3:
                    # assume H, W, N and split channels
                    imgs = [np.asarray(img_stack[..., i]) for i in range(img_stack.shape[2])]
                else:
                    raise ValueError("Unsupported img_stack shape for preview.")

            # Optional titles from metadata if available
            titles = []
            for i, md in enumerate(self.metadata):
                name = getattr(md, "name", None) or getattr(md, "sample_id", None) \
                       or getattr(md, "path", None)
                try:
                    name = Path(name).name  # shorten if it's a path
                except Exception:
                    pass
                titles.append(str(name) if name is not None else f"image_{i}")

            self._preview_images(imgs, titles)

        if self.confirm:
            apply = Confirm.ask("Continue with lobule segmentation?", default=True, console=console)
            if not apply:
                console.print("Aborted by user. No segmentation performed.", style="error")
                return self.metadata

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        new_uid = f"{timestamp}-{short_id}"

        report_path = OUTPUT_PATH / f"segmentation-{new_uid}"
        report_path.mkdir(parents=True, exist_ok=True)

        # Superpixel algorithm (steps 3–7)
        pad = 10  # adjust if you want a different safety border

        # Adjust region size from number of superpixels if specified
        if self.target_superpixels is not None:
            H, W, _ = img_stack.shape
            region_size = int(np.sqrt((H * W) / self.target_superpixels))
            console.print(f"Region size for SLIC was computed as {region_size}", style="info")
            if region_size < self.region_size:
                console.print(f"Computed region size {region_size} is smaller than specified region size {self.region_size}", style="warning")
            self.region_size = region_size

        thinned, (vessel_classes, vessel_contours) = run_skeletize_image(
            img_stack,
            n_clusters=3,
            pad=pad,
            region_size = self.region_size,
            report_path=report_path,  # or a folder path if you want PNGs saved
        )

        console.print("Complete. Creating lines segments from skeleton...", style="info")

        # Steps 8: segments
        line_segments = segment_thinned_image(thinned)

        console.print("Complete. Creating segmentation mask...", style="info")

        # Step 9: Creating lobule and vessel polygons from line segments and vessel contours
        mask = process_segments_to_mask(line_segments, thinned.shape, report_path=report_path)

        # Crop mask by padding
        mask_cropped = mask[pad:-pad, pad:-pad]

        console.print("Complete. Back-mapping mask to all pyramid levels...", style="info")

        # Build masks for every level (steps 1–3)
        mask_pyramid = build_mask_pyramid_from_processed(
            mask_cropped=mask_cropped,
            img_size_base=img_size_base,  # ROI size at base_level (after bbox crop)
            bbox_base=bbox,  # bbox in base_level coords used for the crop
            orig_shapes=orig_shapes,  # {level: (H,W)} from self.load_image(0)
            base_level=self.base_level,
        )

        console.print("Complete.", style="info")

        # TODO: package `full_mask` as a new Metadata and save



if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_paths = [DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch0.tiff",
                   DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch1.tiff",
                   DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch0.tiff",
                   DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch1.tiff",
                   ]

    metadata_for_segmentation = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Segmentor = LobuleSegmentor(metadata_for_segmentation,
                                #1,
                                base_level = 4,
                                region_size = 6)

    metadata_segmentation = Segmentor.apply()
