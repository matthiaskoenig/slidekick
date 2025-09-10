import numpy as np
from pathlib import Path
from rich.prompt import Confirm
from typing import List, Union, Optional, Tuple, Dict, Any
from PIL import Image
import cv2
import datetime
import uuid
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH
from slidekick.io import save_tif

from slidekick.processing.roi.roi_utils import largest_bbox, ensure_grayscale_uint8, crop_image, detect_tissue_mask
from slidekick.processing.lobule_segmentation.get_segments import segment_thinned_image
from slidekick.processing.lobule_segmentation.process_segments import process_segments_to_mask
from slidekick.processing.lobule_segmentation.lob_utils import detect_tissue_mask_multiotsu, overlay_mask, pad_image, build_mask_pyramid_from_processed


class LobuleSegmentor(BaseOperator):
    """
    Based on https://github.com/matthiaskoenig/zonation-image-analysis/blob/develop/src/zia/pipeline/pipeline_components/segementation_component.py
    """

    def __init__(self,
                 metadata: Union[Metadata, List[Metadata]],
                 channel_selection: Union[int, List[int]] = None,
                 channels_pp: Union[int, List[int]] = None,
                 channels_pv: Union[int, List[int]] = None,
                 throw_out_ratio: float = None,
                 preview: bool = True,
                 confirm: bool = True,
                 base_level: int = 0,
                 region_size: int = 20,
                 multi_otsu: bool = True,
                 ksize: int = 7,
                 n_clusters: int = 3,
                 alpha_pv: float = 2.0,
                 alpha_pp: float = 2.0,
                 target_superpixels: int = None):
        """
        @param metadata: List or single metadata object to load and use for lobule segmentation. All objects used should be single channel and either periportal or pericentrally expressed.
        @param channel_selection: Number of Metadata objects that should be inverted. Channels with stronger, i.e., brighter expression / absorpotion perincentrally should be inverted. If None, invert none.
        @param channels_pp: Channel[s] that are brighter in periportal regions (potentially after inversion). At least one channel has to be set for channel_pp or channel_pv. If only one is provided, the other will be calculated as the furthest distance on the same channel(s).
        @param channels_pv: Channel[s] that are brighter in periveneous regions (potentially after inversion). At least one channel has to be set for channel_pp or channel_pv. If only one is provided, the other will be calculated as the furthest distance on the same channel(s).
        @param throw_out_ratio: the min percentage of non_background pixel for the slide to have to not be discarded
        @param preview: whether to preview the segmentation
        @param confirm: whether to confirm the segmentation
        @param base_level: Pyramid level to load
        @param multi_otsu: whether to use multi-otsu to filter out microscopy background, otherwise classic otsu
        @param region_size: average size of superpixels for SLIC in pixels
        @param ksize: kernel size for convolution in filtering (cv2.median blur), must be odd and greater than 1, e.g., 3, 5, 7, ...
        @param n_clusters: number of clusters in (weighted) K-Means to use for superpixel clustering
        @param alpha_pv: weighting factors for superpixel clustering for pv channels
        @param alpha_pp: weighting factors for superpixel clustering for pp channels
        @param target_superpixels: number of superpixels to use for segmentation, overrides region_size
        """

        self.throw_out_ratio = throw_out_ratio
        self.preview = preview
        self.confirm = confirm
        self.base_level = base_level
        self.multi_otsu = multi_otsu
        self.region_size = region_size
        self.ksize = ksize
        self.n_clusters = n_clusters
        self.alpha_pv = alpha_pv
        self.alpha_pp = alpha_pp
        self.target_superpixels = target_superpixels
        # make sure channel is list for later iteration
        if isinstance(channel_selection, int):
            channel_selection = [channel_selection]
        elif channel_selection is None:
            channel_selection = []
        super().__init__(metadata, channel_selection=None)
        # Override init back to have direct info on channels to invert
        self.channels = channel_selection
        # Make lists
        if isinstance(channels_pp, int):
            channels_pp = [channels_pp]
        if isinstance(channels_pv, int):
            channels_pv = [channels_pv]
        # check that the defined channels exist in the metadata objects
        if (
                (channels_pp is not None and any(i > len(metadata) - 1 for i in channels_pp)) or
                (channels_pv is not None and any(i > len(metadata) - 1 for i in channels_pv))
        ):
            console.print(
                "At least one channel for pp or pv is greater than number of metadata objects.",
                style="error"
            )
            raise ValueError(
                "At least one channel for pp or pv is greater than number of metadata objects."
            )
        # Check that at least one of the lists is not None:
        if channels_pp is None and channels_pv is None:
            console.print(f"No values for channel_pp or channel_pv provided.", style="error")
            raise ValueError(f"At least one channel hast to be set in either channel_pp or channel_pv.")
        elif channels_pp is not None and channels_pv is not None:
            # Check that channels_pp and channels_pv are different
            overlap = set(channels_pp) & set(channels_pv)
            if overlap:
                console.print(
                    f"The following IDs are used both for periportal and perivenous detection: {overlap}. This is not possible.",
                    style="error")
                raise ValueError(f"Identical elements for channels_pp and channels_pv: {overlap}")
        self.channels_pp = channels_pp
        self.channels_pv = channels_pv

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

    def _filter(self, image_stack: np.ndarray) -> np.ndarray:
        """
        Channel-wise version of zia's filtering. Works for N >= 1 channels.
        Keeps dtype uint8, returns (H, W, N).
        """

        if image_stack.ndim != 3:
            raise ValueError(f"Expected (H, W, N) stack, got {image_stack.shape}")

        H, W, N = image_stack.shape

        # set missing to 0 (any zero across channels -> zero all channels)
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

            # blur, ksize from zia=7
            ch = cv2.medianBlur(ch, ksize=self.ksize)

            # adaptive histogram norm (2D only)
            ch = clahe.apply(ch)
            ch[ch < 10] = 0  # suppress background altered by CLAHE

            # blur
            ch = cv2.medianBlur(ch, ksize=3)

            # channel normalization to 99th percentile
            # use >0 to avoid zero-inflation from background
            nz = ch[ch > 0]
            p99 = float(np.percentile(nz, 99)) if nz.size else 255.0
            if p99 <= 0:
                p99 = 1.0
            ch = (ch.astype(np.float32) / p99 * 255.0).clip(0, 255).astype(np.uint8)

            out[..., c] = ch

        return out

    def _preview_images(self, images: List[np.ndarray], titles: Optional[List[str]] = None) -> None:
        """
        Display the provided images in a Napari viewer as separate layers.
        Each image will be added as a grayscale or RGBA layer (with appropriate naming).
        Napari is launched only if preview is True.
        """
        if not images:
            console.print("No images to preview.", style="warning")
            return

        # Downsample each image for preview to a reasonable size (max 2048 px on longest side)
        thumbs = [self._downsample_to_max_side(im, 2048) for im in images]

        # Import Napari and create a viewer for layered display
        import napari
        viewer = napari.Viewer()

        # Add each image as a separate layer in the viewer
        for idx, im in enumerate(thumbs):
            # Determine a layer name using provided titles or default naming
            layer_name = titles[idx] if titles and idx < len(titles) else f"Layer_{idx}"
            # Add as grayscale or color layer depending on image shape
            if im.ndim == 2:
                viewer.add_image(im, name=str(layer_name), colormap='gray')
            elif im.ndim == 3 and im.shape[-1] in (3, 4):
                # If image has 3 or 4 channels, treat it as an RGB(A) image
                viewer.add_image(im, name=str(layer_name), rgb=True)
            else:
                # Fallback: add image with default settings
                viewer.add_image(im, name=str(layer_name))

        # Start the Napari event loop (blocks until all viewer windows are closed)
        napari.run()

    def _load_and_invert_images_from_metadatas(self, report_path: Path = None) -> [np.ndarray, Tuple[int, int, int, int], Dict[int, Any]]:
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
        if self.multi_otsu:
            # We detect three levels in each wsi: actual tissue, black background and the background in microscopy.
            # returns tissue_mask
            tissue_mask = detect_tissue_mask_multiotsu(stack_grayscale, morphological_radius=6, report_path=report_path)
        else:
            # just tissue detection
            tissue_mask = detect_tissue_mask(stack_grayscale, morphological_radius=6)

        # stack: (H,W,C); tissue_mask: (H,W) True=tissue
        stack[~tissue_mask] = 0

        # crop to tissue bbox
        bbox = largest_bbox(tissue_mask.astype(np.uint8))
        stack = crop_image(stack, bbox)

        # Inversion and discard only AFTER cropping:
        for i in range(stack.shape[-1]):

            # optional inversion
            if self.channels is not None and i in self.channels:
                if np.issubdtype(stack[:, :, i].dtype, np.integer):
                    info = np.iinfo(stack[:, :, i].dtype)
                    stack[:, :, i] = info.max - stack[:, :, i]
                else:  # float images assumed in [0,1]
                    stack[:, :, i] = 1.0 - stack[:, :, i]

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
                    if i in self.channels_pp or i in self.channels_pv:
                        console.print(
                            f"Channel used for periportal or perivenous detection removed.",
                            style="warning",
                        )
                        if i in self.channels_pp:
                            self.channels_pp.remove(i)
                        else:
                            self.channels_pv.remove(i)

                        # Check if channels_pp and channels_pv are not empty lists now
                        if not self.channels_pp and not self.channels_pv:
                            console.print("No remaining channels for perivenous or periportal detection available.", style="error")
                            raise Exception(f"No remaining channels for segmentation available.")

                    # IDs in channels_pp and channels_pv have to be adapted if channel is removed
                    self.channels_pp = [j-1 for j in self.channels_pp if j >= i]
                    self.channels_pv = [j-1 for j in self.channels_pv if j >= i]

        return stack, bbox, orig_shapes

    def skeletize_image(self, image_stack: np.ndarray,
                        pad=10,
                        region_size=6,
                        report_path: Path = None) -> Tuple[np.ndarray, Tuple[List[int], list]]:
        """
        This function is adopting most of the code from zia/pipeline/pipeline_components/algorithm/segementation/clustering.py
        However, this code does not rely on "empty" holes to find vessels, but uses intensities of pre-defined stain-channels
        For this, we use the same superpixel approach, but instead of looking for holes, we cluster the superpixels into each, and then use the pre-defined channels as key indicator which center is which.
        We check for holes after finding high intensity regions for each and then define this as vessel, otherwise the middle of each intensity peak until sharp increase starts
        """

        # 1) Superpixel generation
        image_stack = pad_image(image_stack, pad)

        console.print("Generating superpixels...", style="info")
        superpixelslic = cv2.ximgproc.createSuperpixelSLIC(image_stack, algorithm=cv2.ximgproc.MSLIC,
                                                           region_size=region_size)
        superpixelslic.iterate(num_iterations=20)

        superpixel_mask = superpixelslic.getLabelContourMask(thick_line=False)
        # Get the labels and number of superpixels
        labels = superpixelslic.getLabels()

        num_labels = superpixelslic.getNumberOfSuperpixels()

        if report_path is not None:
            cv2.imwrite(str(report_path / "superpixels.png"), superpixel_mask)

        merged = image_stack.astype(float)

        super_pixels = {label: merged[labels == label] for label in range(num_labels)}

        # 2) Find foreground / background, background is more than 50% smaller than values of 20
        background_pixels = {}
        foreground_pixels = {}

        console.print("Cluster superpixels into foreground and background pixels", style="info")
        for label, pixels in super_pixels.items():
            # pixels shape (N, C)
            channel_dark_fraction = (pixels <= 0).sum(axis=0) / pixels.shape[0]
            if (channel_dark_fraction > 0.5).any():  # any channel passes
                background_pixels[label] = pixels
            else:
                foreground_pixels[label] = pixels

        # calculate the mean over each channel within each foreground superpixel
        fg_keys = list(foreground_pixels.keys())
        foreground_pixels_means = {
            lab: np.mean(arr, axis=0).astype(np.float32) for lab, arr in foreground_pixels.items()
        }

        if report_path is not None:
            # export a foreground/background visualization over superpixels
            out_template = np.zeros(shape=(image_stack.shape[0], image_stack.shape[1], 3), dtype=np.uint8)
            for i in range(num_labels):
                if i in background_pixels.keys():
                    out_template[labels == i] = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    out_template[labels == i] = np.array([255, 255, 255], dtype=np.uint8)
            cv2.imwrite(str(report_path / "superpixels_bg_fg.png"), out_template)

            # export per-channel superpixel means as images
            for k in range(merged.shape[2]):
                out_gray = np.zeros(shape=(image_stack.shape[0], image_stack.shape[1]), dtype=np.float32)
                for i, mean_vec in foreground_pixels_means.items():
                    out_gray[labels == i] = mean_vec[k]
                m = float(out_gray.max())
                if m > 0:
                    out_gray = (out_gray / m) * 255.0
                out_u8 = out_gray.astype(np.uint8)
                out_rgb = cv2.cvtColor(out_u8, cv2.COLOR_GRAY2RGB)
                mask_on = (np.all(out_rgb != [0, 0, 0], axis=2)) & (superpixel_mask == 255)
                out_rgb[mask_on] = [255, 255, 0]
                cv2.imwrite(str(report_path / f"superpixel_mean_{k}.png"), out_rgb)

            # histograms of superpixel mean intensity per channel
            X_means = np.stack([foreground_pixels_means[k] for k in fg_keys], axis=0)  # [M, C]
            C = X_means.shape[1]
            fig, axes = plt.subplots(
                1, C,
                figsize=(2.8 * C, 3.0),
                squeeze=False,
                constrained_layout=True,
            )
            for c in range(C):
                axes[0, c].hist(X_means[:, c], bins=50)
                axes[0, c].set_title(f"Ch {c} · mean")
                axes[0, c].set_xlabel("value")
                axes[0, c].set_ylabel("count")
            fig.suptitle("Superpixel mean intensity histograms", fontsize=11)
            fig.savefig(str(report_path / "hist_superpixel_means.png"), dpi=150)
            plt.close(fig)

        # 3) cluster the superpixels based on the mean channel values within the superpixel using weighted K-Means
        console.print(f"Cluster (n={self.n_clusters}) the foreground superpixels based on superpixel mean values",
                      style="info")

        # feature matrix [M, C] from superpixel means
        X = np.stack([foreground_pixels_means[k] for k in fg_keys], axis=0)
        C = X.shape[1]

        # per-channel importance: boost PP/PV channels
        w_feat = np.ones(C, dtype=np.float32)
        if self.channels_pp is not None:
            w_feat[self.channels_pp] *= self.alpha_pp
        if self.channels_pv is not None:
            w_feat[self.channels_pv] *= self.alpha_pv

        # scaling implements feature weights
        Xw = X * w_feat

        # KMeans on weighted mean features
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(Xw)

        centers = kmeans.cluster_centers_ / w_feat  # back to original scale
        labels_k = kmeans.labels_

        # assign centers: indices for PP, MID, PV using mean over provided channels
        channels_pp = self.channels_pp
        channels_pv = self.channels_pv
        if (channels_pp is not None) and (channels_pv is not None):
            pp_scores = centers[:, channels_pp].mean(axis=1)
            pv_scores = centers[:, channels_pv].mean(axis=1)
            idx_pp = int(np.argmax(pp_scores))
            pv_order = np.argsort(-pv_scores)
            idx_pv = int(pv_order[0] if pv_order[0] != idx_pp else pv_order[1])
        elif channels_pp is not None:
            s = centers[:, channels_pp].mean(axis=1)
            idx_pp = int(np.argmax(s))
            idx_pv = int(np.argmin(s))  # inverse pole on same channels
        else:
            s = centers[:, channels_pv].mean(axis=1)
            idx_pv = int(np.argmax(s))
            idx_pp = int(np.argmin(s))  # inverse pole on same channels

        idx_mid = [i for i in range(self.n_clusters) if i not in [idx_pp, idx_pv]]
        sorted_label_idx = np.array([idx_pp, *idx_mid, idx_pv], dtype=int)
        
        superpixel_kmeans_map = {sp: km for sp, km in zip(foreground_pixels.keys(), labels_k)}
        lookup = np.vectorize(superpixel_kmeans_map.get)
        foreground_clustered = lookup(labels)

        # TODO: Vessel classfication through superpixels over superpixel

        # create template for the background / vessels for classification
        console.print("Complete. Create hierarchical grayscale image of clusters...", style="info")
        background_template = np.ones_like(merged[:, :, 0]) * 255
        for i in range(self.n_clusters):
            background_template[foreground_clustered == sorted_label_idx[i]] = 0

        template = np.zeros(shape=(image_stack.shape[0], image_stack.shape[1])).astype(np.uint8)
        for i in range(self.n_clusters):
            if i == 0:
                n = 1
            elif i == self.n_clusters - 1:
                n = 3
            else:
                n = 2
            template[foreground_clustered == sorted_label_idx[i]] = round(n * 255 / 3)

        if report_path is not None:
            cv2.imwrite(str(report_path / "foreground_clustered.png"), template)

        console.print("Complete. Run thinning algorithm...", style="info")

        # TODO: Draw final map + vessels

        template = cv2.medianBlur(255 - template, 5)

        console.print("Complete. Run thinning algorithm...", style="info")

        thinned = cv2.ximgproc.thinning(template.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

        # TODO: Double thinning + vessels

        if report_path is not None:
            cv2.imwrite(str(report_path / "thinned.png"), thinned)

        return thinned, (None, None)  #thinned, (classes, filtered_contours)


    def apply(self) -> Metadata:

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        new_uid = f"{timestamp}-{short_id}"

        report_path = OUTPUT_PATH / f"segmentation-{new_uid}"
        report_path.mkdir(parents=True, exist_ok=True)

        console.print("Loading images...", style="info")
        # Load images and invert based on metadata
        img_stack, bbox, orig_shapes = self._load_and_invert_images_from_metadatas(report_path)

        img_size_base = img_stack.shape[:2]  # base size for back-mapping of mask

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

        thinned, (vessel_classes, vessel_contours) = self.skeletize_image(
            img_stack,
            pad=pad,
            region_size=self.region_size,
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

        # Preview
        # Left: Orig, Middle: Overlay, Right: Mask only
        if self.preview:
            # Original as mean projection
            orig_vis = img_stack.mean(axis=2).astype(np.uint8)
            # Skeleton cropped to match mask/img_stack
            overlay_vis = overlay_mask(img_stack, mask_cropped, alpha=0.5)
            # Mask visualization as binary 0/255 for clarity
            mask_vis = (mask_cropped > 0).astype(np.uint8) * 255
            self._preview_images([orig_vis, overlay_vis, mask_vis], titles=["Original", "Segmentation Overlay", "Mask"])

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

        new_meta = Metadata(
            path_original=report_path,
            path_storage=report_path,
            image_type="mask",
            uid=new_uid,
        )

        new_meta.save(report_path)
        # Pass the full level -> array dict so save_tif writes a tiled, pyramidal TIFF
        save_tif(mask_pyramid, report_path / f"{new_uid}_seg.tiff", metadata=new_meta)

        return new_meta



if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_paths = [DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch0.tiff",  # pv
                   #DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch1.tiff",
                   #DATA_PATH / "reg_n_sep" / "GS_CYP1A2_ch2.tiff", # -> DAPI
                   DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch0.tiff",  # pp
                   #DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch1.tiff",
                   #DATA_PATH / "reg_n_sep" / "Ecad_CYP2E1_ch2.tiff", # -> DAPI
                   ]

    metadata_for_segmentation = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Segmentor = LobuleSegmentor(metadata_for_segmentation,
                                #channels_pp=1,
                                channels_pv=0,
                                base_level=4,
                                region_size=25)

    metadata_segmentation = Segmentor.apply()
