import numpy as np
import os
from pathlib import Path
from rich.prompt import Confirm, Prompt
import zarr
from typing import List, Union, Optional
from PIL import Image
import cv2
import datetime
import uuid

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH

from slidekick.processing.lobule_segmentation.run_skeletize_image import run_skeletize_image

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
                 base_level: int = 0):
        """
        @param metadata: List or single metadata object to load and use for lobule segmentation
        @param channel_selection: Number of Metadata objects that should be inverted. If None, invert none.
        Images should be inverted so that high absorbance, i.e., bright values are pericentral zones.
        Different to zia where simply high absorbance = bright
        @param throw_out_ratio: the min percentage of non_background pixel for the slide to have to not be discarded
        @param preview: whether to preview the segmentation
        @param confirm: whether to confirm the segmentation
        @param base_level: Pyramid level to load
        """
        self.throw_out_ratio = throw_out_ratio
        self.preview = preview
        self.confirm = confirm
        self.base_level = base_level
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

            # optional discard by foreground coverage
            if self.throw_out_ratio is not None:
                non_bg = np.count_nonzero(img != 255)
                ratio = non_bg / img.size
                if ratio < self.throw_out_ratio:
                    console.print(
                        f"Discarded {i} with non-background pixel ratio: {ratio:.3f}",
                        style="warning",
                    )
                    continue

            # optional inversion
            if self.channels is not None and i in self.channels:
                if np.issubdtype(img.dtype, np.integer):
                    info = np.iinfo(img.dtype)
                    img = info.max - img
                else:  # float images assumed in [0,1]
                    img = 1.0 - img

            arrays.append(img)

        if not arrays:
            raise RuntimeError("No images loaded after filtering.")

        # stack along the third dimension
        # harmonize shapes before stacking
        h_min = min(a.shape[0] for a in arrays)
        w_min = min(a.shape[1] for a in arrays)
        arrays = [a[:h_min, :w_min] for a in arrays]

        stack = np.dstack(arrays).astype(np.uint8)

        return stack

    def _filter(self, image_stack: np.ndarray) -> np.ndarray:
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

    def _overlay_skeleton(self, image_stack: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        """Red overlay of skeleton on mean of channels."""
        base = image_stack.mean(axis=2).astype(np.uint8)
        rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        rgb[skeleton > 0] = (255, 0, 0)
        return rgb

    def apply(self) -> Metadata:

        console.print("Loading images...", style="info")
        # Load images and invert based on metadata
        img_stack = self._load_and_invert_images_from_metadatas()

        console.print("Complete. Now applying filters...", style="info")

        # Apply filters
        img_stack = self._filter(img_stack)

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
        pad = 0  # adjust if you want a safety border
        thinned, (vessel_classes, vessel_contours) = run_skeletize_image(
            img_stack,
            n_clusters=3,
            pad=pad,
            report_path=report_path,  # or a folder path if you want PNGs saved
        )




if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_paths = [DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch0.tiff",
                   DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch1.tiff",
                   DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch2.tiff",
                   DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch0.tiff",
                   DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch1.tiff",
                   DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch2.tiff",
                   ]

    metadata_for_segmentation = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Segmentor = LobuleSegmentor(metadata_for_segmentation, base_level=2)

    metadata_segmentation = Segmentor.apply()
