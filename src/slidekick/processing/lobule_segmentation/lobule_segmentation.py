import numpy as np
from pathlib import Path
from rich.prompt import Confirm
from typing import List, Union, Optional, Tuple, Dict, Any
import cv2
import datetime
import uuid
from sklearn.cluster import KMeans
import napari

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from magicgui import magicgui
from magicgui.widgets import Container, PushButton

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH
from slidekick.io import save_tif

from slidekick.processing.roi.roi_utils import largest_bbox, ensure_grayscale_uint8, crop_image, detect_tissue_mask
from slidekick.processing.lobule_segmentation.get_segments import segment_thinned_image
from slidekick.processing.lobule_segmentation.segments_to_mask import process_segments_to_mask
from slidekick.processing.lobule_segmentation.portality import mask_to_portality
from slidekick.processing.lobule_segmentation.lob_utils import (
    detect_tissue_mask_multiotsu, overlay_mask, pad_image, build_mask_pyramid_from_processed,
    downsample_to_max_side, render_cluster_gray, nonlinear_channel_weighting, to_base_full, rescale_full
)

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
                 target_superpixels: int = None,
                 adaptive_histonorm: bool = True,
                 # nonlinear KMeans
                 interactive_weighting: bool = True,
                 nonlinear_kmeans: bool = True,
                 alpha_pv: float = 2.0,
                 alpha_pp: float = 3.0,
                 pp_gamma: float = 0.65,
                 pv_gamma: float = 0.75,
                 nl_low_pct: float = 5.0,
                 nl_high_pct: float = 90.0,
                 # vessel gating
                 interactive_vessels: bool = True,
                 min_vessel_area_pp: int = 200,
                 min_vessel_area_pv: int = 200,
                 vessel_annulus_px: int = 5,
                 vessel_zone_ratio_thr_pp: float = 0.35,
                 vessel_zone_ratio_thr_pv: float = 0.55,
                 vessel_circularity_min: float = 0.12,
                 min_area_px: int = 5000):
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
        @param target_superpixels: number of superpixels to use for segmentation, overrides region_size
        @param adaptive_histonorm: use adaptive histogram norm in filtering
        @param interactive_weighting: whether to use interactive weighting in a napari preview
        @param nonlinear_kmeans: use a non-linear feature weighting in K_means
        @param alpha_pv: weighting factors for superpixel clustering for pv channels in linear KMeans
        @param alpha_pp: weighting factors for superpixel clustering for pp channels in linear KMeans
        @param pp_gamma: Gamma exponent applied to PP channels after robust percentile normalization. smaller = stronger boost
        @param pv_gamma: Gamma exponent applied to PV channels after robust percentile normalization. smaller = stronger boost
        @param nl_low_pct: Lower percentile used as the per-channel floor before gamma.
        @param nl_high_pct: Upper percentile used as the per-channel ceiling before gamma.
        @param interactive_vessels: whether to use interactive vessel detection/grouping in a napari preview
        @param min_vessel_area_pp: min area (px) for PP-enclosed vessel candidates
        @param min_vessel_area_pv: min area (px) for PV-enclosed vessel candidates
        @param vessel_annulus_px: thickness (px) of ring-consistency check
        @param vessel_zone_ratio_thr_pp: fraction of ring that must be PP
        @param vessel_zone_ratio_thr_pv: fraction of ring that must be PV
        @param vessel_circularity_min: 4πA/P^2 gate; low keeps elongated vessels
        @param min_area_px: Define area threshold to filter out very small polygons (noise)
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
        self.adaptive_histonorm = adaptive_histonorm

        # Kmeans
        self.interactive_weighting = interactive_weighting
        self.nonlinear_kmeans = nonlinear_kmeans
        self.pp_gamma = pp_gamma
        self.pv_gamma = pv_gamma
        self.nl_low_pct = nl_low_pct
        self.nl_high_pct = nl_high_pct

        # vessel gating
        self.interactive_vessels = interactive_vessels
        self.min_vessel_area_pp = min_vessel_area_pp
        self.min_vessel_area_pv = min_vessel_area_pv
        self.vessel_annulus_px = vessel_annulus_px
        self.vessel_zone_ratio_thr_pp = vessel_zone_ratio_thr_pp
        self.vessel_zone_ratio_thr_pv = vessel_zone_ratio_thr_pv
        self.vessel_circularity_min = vessel_circularity_min

        # lobule generation
        self.min_area_px = min_area_px

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
            if self.adaptive_histonorm:
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
        thumbs = [downsample_to_max_side(im, 2048) for im in images]

        # Import Napari and create a viewer for layered display
        # TODO: Use viewer from slidekick
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

    def _preview_channel_weighting(
            self,
            image_stack: np.ndarray,
    ) -> None:
        """
        Interactive Napari preview that shows how the KMeans / nonlinear
        weighting parameters affect the *composite* PP and PV channels
        BEFORE superpixel generation and KMeans.

        - Uses a downsampled copy of `image_stack` for speed.
        - Computes one weighted PP image and one weighted PV image
          (mean over the respective channels after nonlinear+alpha weights).
        - Updates overlays automatically when a control changes (no 'Update' button).
        - Provides 'Reset to defaults' and 'Confirm and continue' buttons.
        """
        if image_stack.ndim != 3:
            raise ValueError(f"Expected (H, W, C) stack, got {image_stack.shape}")

        # only show relevant channels (PP + PV)
        channels_interest: List[int] = []
        if self.channels_pp is not None:
            channels_interest.extend(self.channels_pp)
        if self.channels_pv is not None:
            channels_interest.extend(self.channels_pv)
        channels_interest = sorted(set(channels_interest))

        if not channels_interest:
            console.print(
                "Interactive weighting requested, but no channels_pp/channels_pv defined.",
                style="warning",
            )
            return

        # downsample whole stack for preview
        stack_small = downsample_to_max_side(image_stack, 1024)
        H, W, C = stack_small.shape

        # remember current values as "defaults" for reset
        defaults = dict(
            nonlinear_kmeans=self.nonlinear_kmeans,
            alpha_pp=self.alpha_pp,
            alpha_pv=self.alpha_pv,
            pp_gamma=self.pp_gamma,
            pv_gamma=self.pv_gamma,
            nl_low_pct=self.nl_low_pct,
            nl_high_pct=self.nl_high_pct,
        )

        def compute_pp_pv_images() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            """
            Compute composite PP and PV images that mirror the feature
            transformation used for KMeans:

              - optional nonlinear_channel_weighting (gamma + percentiles)
              - alpha_pp / alpha_pv linear weighting

            Returns (pp_img, pv_img) as uint8 arrays in [0, 255],
            or None if the respective channel list is empty.
            """
            flat = stack_small.reshape(-1, C).astype(np.float32)

            # Nonlinear weighting (same as in skeletize_kmeans, but on pixels)
            if self.nonlinear_kmeans:
                X = nonlinear_channel_weighting(
                    flat,
                    self.channels_pp,
                    self.channels_pv,
                    self.pp_gamma,
                    self.pv_gamma,
                    self.nl_low_pct,
                    self.nl_high_pct,
                )
            else:
                X = flat

            # Linear weights alpha_pp / alpha_pv per feature
            w_feat = np.ones(X.shape[1], dtype=np.float32)
            if self.channels_pp:
                for idx in self.channels_pp:
                    if 0 <= idx < w_feat.shape[0]:
                        w_feat[idx] *= self.alpha_pp
            if self.channels_pv:
                for idx in self.channels_pv:
                    if 0 <= idx < w_feat.shape[0]:
                        w_feat[idx] *= self.alpha_pv

            Xw = X * w_feat
            Xw_img = Xw.reshape(H, W, C)

            pp_img: Optional[np.ndarray] = None
            pv_img: Optional[np.ndarray] = None

            # composite PP: mean over weighted PP channels, then normalize 0–255
            if self.channels_pp:
                pp_vals = Xw_img[..., self.channels_pp]
                if pp_vals.ndim == 2:
                    pp_float = pp_vals
                else:
                    pp_float = pp_vals.mean(axis=2)
                mn = float(np.nanmin(pp_float))
                mx = float(np.nanmax(pp_float))
                if mx > mn:
                    pp_img = (((pp_float - mn) / (mx - mn)) * 255.0).astype(np.uint8)
                else:
                    pp_img = np.zeros((H, W), dtype=np.uint8)

            # composite PV: mean over weighted PV channels, then normalize 0–255
            if self.channels_pv:
                pv_vals = Xw_img[..., self.channels_pv]
                if pv_vals.ndim == 2:
                    pv_float = pv_vals
                else:
                    pv_float = pv_vals.mean(axis=2)
                mn = float(np.nanmin(pv_float))
                mx = float(np.nanmax(pv_float))
                if mx > mn:
                    pv_img = (((pv_float - mn) / (mx - mn)) * 255.0).astype(np.uint8)
                else:
                    pv_img = np.zeros((H, W), dtype=np.uint8)

            return pp_img, pv_img

        pp_img, pv_img = compute_pp_pv_images()

        # Napari viewer
        viewer = napari.Viewer()
        # TODO: Integrate with slidekick viewer

        pp_layer = None
        pv_layer = None

        if pp_img is not None:
            pp_layer = viewer.add_image(
                pp_img,
                name="PP channel (weighted)",
                colormap="green",
                blending="additive",
                opacity=0.5,
            )

        if pv_img is not None:
            pv_layer = viewer.add_image(
                pv_img,
                name="PV channel (weighted)",
                colormap="magenta",
                blending="additive",
                opacity=0.5,
            )

        @magicgui(
            layout="vertical",
            auto_call=True,  # update on every change
            nonlinear_kmeans={"widget_type": "CheckBox"},
            alpha_pp={"min": 0.0, "max": 5.0, "step": 0.1},
            alpha_pv={"min": 0.0, "max": 5.0, "step": 0.1},
            pp_gamma={"min": 0.1, "max": 2.0, "step": 0.05},
            pv_gamma={"min": 0.1, "max": 2.0, "step": 0.05},
            nl_low_pct={"min": 0.0, "max": 20.0, "step": 1.0},
            nl_high_pct={"min": 50.0, "max": 100.0, "step": 1.0},
        )
        def weighting_controls(
                nonlinear_kmeans: bool = self.nonlinear_kmeans,
                alpha_pp: float = self.alpha_pp,
                alpha_pv: float = self.alpha_pv,
                pp_gamma: float = self.pp_gamma,
                pv_gamma: float = self.pv_gamma,
                nl_low_pct: float = self.nl_low_pct,
                nl_high_pct: float = self.nl_high_pct,
        ):
            """
            This function is called automatically whenever any control changes.
            It updates self.*, recomputes weighted PP/PV images and updates layers.
            """
            # update object state: these will be used later by skeletize_kmeans
            self.nonlinear_kmeans = nonlinear_kmeans
            self.alpha_pp = alpha_pp
            self.alpha_pv = alpha_pv
            self.pp_gamma = pp_gamma
            self.pv_gamma = pv_gamma
            self.nl_low_pct = nl_low_pct
            self.nl_high_pct = nl_high_pct

            console.print(
                f"Updated KMeans weighting params:\n"
                f"  nonlinear_kmeans={self.nonlinear_kmeans}\n"
                f"  alpha_pp={self.alpha_pp}, alpha_pv={self.alpha_pv}\n"
                f"  pp_gamma={self.pp_gamma}, pv_gamma={self.pv_gamma}\n"
                f"  nl_low_pct={self.nl_low_pct}, nl_high_pct={self.nl_high_pct}",
                style="info",
            )

            # recompute composite PP / PV images and update layers
            new_pp, new_pv = compute_pp_pv_images()
            nonlocal pp_layer, pv_layer
            if pp_layer is not None and new_pp is not None:
                pp_layer.data = new_pp
            if pv_layer is not None and new_pv is not None:
                pv_layer.data = new_pv

        # extra buttons: reset to defaults, confirm and continue
        reset_btn = PushButton(text="Reset to defaults")
        confirm_btn = PushButton(text="Confirm and continue")

        def on_reset(*args):
            # restore defaults in the object
            self.nonlinear_kmeans = defaults["nonlinear_kmeans"]
            self.alpha_pp = defaults["alpha_pp"]
            self.alpha_pv = defaults["alpha_pv"]
            self.pp_gamma = defaults["pp_gamma"]
            self.pv_gamma = defaults["pv_gamma"]
            self.nl_low_pct = defaults["nl_low_pct"]
            self.nl_high_pct = defaults["nl_high_pct"]

            # restore defaults in the GUI; auto_call will recompute overlays
            weighting_controls.nonlinear_kmeans.value = defaults["nonlinear_kmeans"]
            weighting_controls.alpha_pp.value = defaults["alpha_pp"]
            weighting_controls.alpha_pv.value = defaults["alpha_pv"]
            weighting_controls.pp_gamma.value = defaults["pp_gamma"]
            weighting_controls.pv_gamma.value = defaults["pv_gamma"]
            weighting_controls.nl_low_pct.value = defaults["nl_low_pct"]
            weighting_controls.nl_high_pct.value = defaults["nl_high_pct"]

        def on_confirm(*args):
            # just close the viewer; napari.run() will return and apply() continues
            viewer.close()

        reset_btn.changed.connect(on_reset)
        confirm_btn.changed.connect(on_confirm)

        controls = Container(widgets=[weighting_controls, reset_btn, confirm_btn], layout="vertical")
        viewer.window.add_dock_widget(controls, area="right")

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


    def skeletize_kmeans(self, image_stack: np.ndarray,
                         pad=10,
                         region_size=6,
                         report_path: Path = None) -> Tuple[np.ndarray, Tuple[List[int], list]]:
        """
        Adopts most code from zia.../clustering.py. Uses intensities of pre-defined stain-channels.
        We cluster superpixels, label centers (PP/MID/PV), then search for ring-like PP/PV regions
        and reassign enclosed superpixels as vessels to the surrounding class. Saves zonation overlay
        with vessel outlines (central=cyan, portal=magenta), then thinning. Adds refinement:
        distance-based filling and histogram cutoffs guided by vessels.
        """
        # 1) Superpixel generation
        image_stack = pad_image(image_stack, pad)

        console.print("Generating superpixels...", style="info")
        superpixelslic = cv2.ximgproc.createSuperpixelSLIC(
            image_stack, algorithm=cv2.ximgproc.MSLIC, region_size=region_size
        )
        superpixelslic.iterate(num_iterations=10)

        superpixel_mask = superpixelslic.getLabelContourMask(thick_line=False)
        labels = superpixelslic.getLabels()
        num_labels = superpixelslic.getNumberOfSuperpixels()

        if report_path is not None:
            cv2.imwrite(str(report_path / "superpixels.png"), superpixel_mask)

        merged = image_stack.astype(float)

        # Vectorized per-label stats (fast)
        H, W, C = image_stack.shape
        lab_flat = labels.ravel()
        img_flat = merged.reshape(-1, C).astype(np.float32)

        counts = np.bincount(lab_flat, minlength=num_labels).astype(np.int32)
        nz = counts > 0

        dark_flat = (img_flat <= 0).astype(np.uint8)
        dark_counts = np.vstack([
            np.bincount(lab_flat, weights=dark_flat[:, c], minlength=num_labels)
            for c in range(C)
        ]).T
        dark_frac = np.divide(dark_counts, counts[:, None], where=nz[:, None])

        fg_label_mask = ~(dark_frac > 0.5).any(axis=1)
        fg_labels = np.nonzero(fg_label_mask)[0]

        fg_pix_mask = np.isin(labels, fg_labels)

        sums = np.vstack([
            np.bincount(lab_flat, weights=img_flat[:, c], minlength=num_labels)
            for c in range(C)
        ]).T
        means = np.zeros_like(sums, dtype=np.float32)
        means[nz] = sums[nz] / counts[nz, None]

        X_means = means[fg_labels]
        fg_keys = fg_labels.tolist()

        if report_path is not None:
            out_template = np.zeros((H, W, 3), dtype=np.uint8)
            out_template[fg_pix_mask] = (255, 255, 255)
            cv2.imwrite(str(report_path / "superpixels_bg_fg.png"), out_template)

            for k in range(C):
                out_gray = means[:, k][labels]
                out_gray[~fg_pix_mask] = 0
                m = float(out_gray.max())
                if m > 0:
                    out_gray = (out_gray / m) * 255.0
                out_u8 = out_gray.astype(np.uint8)
                out_rgb = cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)
                mask_on = (out_u8 > 0) & (superpixel_mask == 255)
                out_rgb[mask_on] = [255, 255, 0]
                cv2.imwrite(str(report_path / f"superpixel_mean_{k}.png"), out_rgb)

            C_ = X_means.shape[1]
            fig, axes = plt.subplots(1, C_, figsize=(2.8 * C_, 3.0), squeeze=False, constrained_layout=True)
            for c in range(C_):
                axes[0, c].hist(X_means[:, c], bins=50)
                axes[0, c].set_title(f"Ch {c} · mean")
                axes[0, c].set_xlabel("value");
                axes[0, c].set_ylabel("count")
            fig.suptitle("Superpixel mean intensity histograms", fontsize=11)
            fig.savefig(str(report_path / "hist_superpixel_means.png"), dpi=150)
            plt.close(fig)

        # 2) Weighted KMeans over superpixel mean features
        console.print(f"Complete. Cluster (n={self.n_clusters}) the foreground superpixels based on superpixel mean values...",
                      style="info")

        if self.nonlinear_kmeans:
            X = X_means.copy()
            C = X.shape[1]

            # non-linear lifting
            X = nonlinear_channel_weighting(
                X,
                self.channels_pp,
                self.channels_pv,
                self.pp_gamma,
                self.pv_gamma,
                self.nl_low_pct,
                self.nl_high_pct,
            )

            # linear weights
            w_feat = np.ones(C, dtype=np.float32)
            if self.channels_pp is not None:
                w_feat[self.channels_pp] *= self.alpha_pp
            if self.channels_pv is not None:
                w_feat[self.channels_pv] *= self.alpha_pv
            Xw = X * w_feat

            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            kmeans.fit(Xw)

        else:

            X = X_means
            C = X.shape[1]

            w_feat = np.ones(C, dtype=np.float32)
            if self.channels_pp is not None:
                w_feat[self.channels_pp] *= self.alpha_pp
            if self.channels_pv is not None:
                w_feat[self.channels_pv] *= self.alpha_pv
            Xw = X * w_feat

            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            kmeans.fit(Xw)

        centers = kmeans.cluster_centers_ / w_feat
        labels_k = kmeans.labels_

        # 3) Assign semantic roles PP/MID/PV
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
            idx_pv = int(np.argmin(s))
        else:
            s = centers[:, channels_pv].mean(axis=1)
            idx_pv = int(np.argmax(s))
            idx_pp = int(np.argmin(s))

        idx_mid = [i for i in range(self.n_clusters) if i not in [idx_pp, idx_pv]]
        sorted_label_idx = np.array([idx_pp, *idx_mid, idx_pv], dtype=int)

        # Map each foreground superpixel id -> cluster id
        superpixel_kmeans_map = {sp: km for sp, km in zip(fg_keys, labels_k)}

        # Build initial assignment dict for all superpixels (background: -1)
        assigned_by_sp: Dict[int, int] = {sp: -1 for sp in range(num_labels)}
        for sp, km in superpixel_kmeans_map.items():
            assigned_by_sp[sp] = km

        # Vectorized lookup of assigned_by_sp over the SLIC label image
        get_lab = np.vectorize(lambda sp: assigned_by_sp.get(int(sp), -1))
        cluster_map = get_lab(labels).astype(np.int32)  # shape (H, W), values in {-1, 0..n_clusters-1}

        if report_path is not None:
            template_init = render_cluster_gray(cluster_map, sorted_label_idx, self.n_clusters)
            template_init[~fg_pix_mask] = 0
            cv2.imwrite(str(report_path / "foreground_clustered.png"), template_init)

        console.print("Complete. Now detecting vessels in detected zonation...", style="info")

        # 4) Vessel detection and superpixel reassignment
        vessel_classes: List[int] = []
        vessel_contours: List[np.ndarray] = []

        k = max(1, int(self.vessel_annulus_px))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

        tissue_bool = (image_stack.mean(axis=2) > 0)
        bg_bool = ~tissue_bool

        def _collect_candidates_from_mask(mask_u8: np.ndarray) -> List[
            Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
            """
            Returns list of (hole_bool, ring_bool, contour, area, circularity)
            from BOTH closed holes and edge-truncated arcs synthesized via erosion.
            """
            out = []
            contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if contours is None or hierarchy is None:
                return out
            Hh = hierarchy[0]

            # permissive minima for candidate gen (final class later)
            min_area = min(int(self.min_vessel_area_pp), int(self.min_vessel_area_pv))
            circ_min = float(self.vessel_circularity_min) * 0.9

            for i, cnt in enumerate(contours):
                # A) CLOSED HOLES (child contours)
                if Hh[i][3] != -1:
                    area = cv2.contourArea(cnt)
                    if area < max(1, min_area):
                        continue
                    peri = max(cv2.arcLength(cnt, True), 1e-6)
                    circ = float(4.0 * np.pi * area / (peri * peri))
                    if circ < circ_min:
                        continue

                    hole_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.drawContours(hole_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    hole_bool = hole_mask > 0

                    dil = cv2.dilate(hole_mask, se)
                    ero = cv2.erode(hole_mask, se)
                    ring = cv2.subtract(dil, ero)
                    ring_bool = ring > 0
                    if not np.any(ring_bool):
                        continue

                    out.append((hole_bool, ring_bool, cnt, area, circ))
                    continue

                # B) EDGE / TRUNCATED BLOBS (external contours touching border)
                x, y, w2, h2 = cv2.boundingRect(cnt)
                touches_border = (x == 0) or (y == 0) or (x + w2 == W) or (y + h2 == H)
                if not touches_border:
                    continue

                area_ext = cv2.contourArea(cnt)
                if area_ext < max(1, min_area):
                    continue
                peri_ext = max(cv2.arcLength(cnt, True), 1e-6)
                circ_ext = float(4.0 * np.pi * area_ext / (peri_ext * peri_ext))
                if circ_ext < circ_min:
                    continue

                ext_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(ext_mask, [cnt], -1, 255, thickness=cv2.FILLED)

                # ensure adjacency to background -> “open” ring
                border_contact = cv2.dilate((bg_bool.astype(np.uint8) * 255), se)
                if float(np.mean((border_contact > 0)[ext_mask > 0])) < 0.10:
                    continue

                inner = cv2.erode(ext_mask, se)
                synth_hole = cv2.erode(inner, se)
                ring_e = cv2.subtract(inner, synth_hole)

                hole_bool = synth_hole > 0
                ring_bool = ring_e > 0
                if not np.any(hole_bool) or not np.any(ring_bool):
                    continue

                out.append((hole_bool, ring_bool, cnt, area_ext, circ_ext))

            return out

        # unified candidate pool from all labeled tissue (avoid PP↔PV bias)
        mask_all = np.zeros((H, W), dtype=np.uint8)
        mask_all[(cluster_map >= 0)] = 255
        candidates = _collect_candidates_from_mask(mask_all)

        # classify by ring-majority; collect (no reassignment yet); then group & reassign
        candidates.sort(key=lambda t: t[3], reverse=True)  # largest first

        # We'll keep full info so we can merge later:
        kept_items: List[Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]] = []

        for hole_bool, ring_bool, cnt, area_c, circ_c in candidates:
            # MID-aware classification (exclude MID & BG from denominator)
            ring_is_mid = np.isin(cluster_map, idx_mid)
            ring_mid = ring_bool & ring_is_mid
            ring_bg = ring_bool & (cluster_map < 0)
            ring_eligible = ring_bool & ~(ring_mid | ring_bg)
            eligible_n = int(np.count_nonzero(ring_eligible))
            if eligible_n == 0:
                continue

            pp_count = int(np.count_nonzero(ring_eligible & (cluster_map == idx_pp)))
            pv_count = int(np.count_nonzero(ring_eligible & (cluster_map == idx_pv)))
            pp_frac = pp_count / float(eligible_n)
            pv_frac = pv_count / float(eligible_n)

            pp_pass = pp_frac >= float(self.vessel_zone_ratio_thr_pp)
            pv_pass = pv_frac >= float(self.vessel_zone_ratio_thr_pv)
            if not (pp_pass or pv_pass):
                continue

            if pp_pass and (not pv_pass or pp_frac >= pv_frac):
                cls = 1
                zone_idx = idx_pp
            else:
                cls = 0
                zone_idx = idx_pv

            # No nesting (centroid inside an existing kept contour of ANY class)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]);
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w2, h2 = cv2.boundingRect(cnt)
                cx, cy = x + w2 // 2, y + h2 // 2

            nested = False
            for (_h, prev_cnt, _c, _z, _r) in kept_items:
                if cv2.pointPolygonTest(prev_cnt, (float(cx), float(cy)), False) >= 0:
                    nested = True
                    break
            if nested:
                continue

            kept_items.append((hole_bool, cnt, cls, zone_idx, ring_bool))

        # Commit: reassign SPs for each detected vessel (no grouping)
        for hole_bool, cnt, cls, zone_idx, _ring_bool in kept_items:
            sp_inside = np.unique(labels[hole_bool])
            for sp in sp_inside:
                assigned_by_sp[int(sp)] = zone_idx
            vessel_contours.append(cnt)
            vessel_classes.append(cls)

        # Rebuild final per-pixel cluster map after reassignment
        cluster_map_final = get_lab(labels).astype(np.int32)

        template = render_cluster_gray(cluster_map_final, sorted_label_idx, self.n_clusters)
        template[~fg_pix_mask] = 0

        # Zonation + vessel overlay
        if report_path is not None:

            base_vis = image_stack.mean(axis=2).astype(np.uint8)
            base_rgb = cv2.cvtColor(base_vis, cv2.COLOR_GRAY2BGR)

            COLOR_PP = (255, 0, 255)
            COLOR_MID = (60, 160, 60)
            COLOR_PV = (0, 165, 255)

            zonation_rgb = np.zeros_like(base_rgb, dtype=np.uint8)
            mask_pp = (cluster_map_final == idx_pp) & fg_pix_mask
            mask_pv = (cluster_map_final == idx_pv) & fg_pix_mask
            mask_mid = np.zeros_like(mask_pp, dtype=bool)
            for mid_idx in idx_mid:
                mask_mid |= (cluster_map_final == mid_idx)
            mask_mid &= fg_pix_mask

            zonation_rgb[mask_pp] = COLOR_PP
            zonation_rgb[mask_mid] = COLOR_MID
            zonation_rgb[mask_pv] = COLOR_PV

            overlay = cv2.addWeighted(base_rgb, 0.6, zonation_rgb, 0.4, 0.0)

            # outlines always on: central=cyan, portal=magenta
            for cnt, cls in zip(vessel_contours, vessel_classes):
                color = (255, 255, 0) if cls == 0 else (255, 0, 255)
                cv2.drawContours(overlay, [cnt], -1, color, thickness=2)

            cv2.imwrite(str(report_path / "zonation_overlay.png"), overlay)
            cv2.imwrite(str(report_path / "zonation_rgb.png"), zonation_rgb)

        # approximate tissue outline from current stack: largest external contour of nonzero region
        tissue_u8 = (tissue_bool.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(tissue_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tissue_boundary = max(cnts, key=cv2.contourArea) if cnts else np.array(
            [[[0, 0]], [[W - 1, 0]], [[W - 1, H - 1]], [[0, H - 1]]], dtype=np.int32
        )

        # FROM HERE: ZIA again
        # shades of gray, n clusters + 2 for background

        class_0_contours = [cnt for cnt, class_ in zip(vessel_contours, vessel_classes) if class_ == 0]
        class_1_contours = [cnt for cnt, class_ in zip(vessel_contours, vessel_classes) if class_ == 1]

        cv2.drawContours(template, class_0_contours, -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(template, class_1_contours, -1, 0, thickness=cv2.FILLED)

        template = 255 - template

        template = cv2.medianBlur(template, 5)

        if report_path is not None:
            cv2.imwrite(str(report_path / "grayscale.png"), template)

        tissue_mask = np.zeros_like(template, dtype=np.uint8)
        cv2.drawContours(tissue_mask, [tissue_boundary], -1, 255, thickness=cv2.FILLED)
        tissue_mask = tissue_mask.astype(bool)

        template[~tissue_mask] = 0
        cv2.drawContours(template, [tissue_boundary], -1, 255, thickness=1)

        if report_path is not None:
            out_template = np.zeros(shape=(merged.shape[0], merged.shape[1], 4)).astype(np.uint8)
            cv2.drawContours(out_template, class_0_contours, -1, (255, 255, 0, 127), thickness=cv2.FILLED)
            cv2.drawContours(out_template, class_0_contours, -1, (255, 255, 0, 255), thickness=2)

            cv2.drawContours(out_template, class_1_contours, -1, (255, 0, 255, 127), thickness=cv2.FILLED)
            cv2.drawContours(out_template, class_1_contours, -1, (255, 0, 255, 255), thickness=2)

            cv2.drawContours(out_template, [tissue_boundary], -1, (255, 255, 255, 255), thickness=3)

            cv2.imwrite(str(report_path / f"classified_vessels.png"), out_template)
            cv2.imwrite(str(report_path / f"final_clustered_map.png"), template)

        console.print("Complete. Run thinning algorithm...", style="info")
        thinned = cv2.ximgproc.thinning(template.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

        # drawing the vessels on the mask and thin again to prevent pixel accumulations the segmentation can't handle

        #cv2.drawContours(thinned, class_0_contours, -1, 0, thickness=cv2.FILLED)
        #cv2.drawContours(thinned, class_0_contours, -1, 255, thickness=1)

        #cv2.drawContours(thinned, class_1_contours, -1, 0, thickness=cv2.FILLED)
        #cv2.drawContours(thinned, class_1_contours, -1, 255, thickness=1, )

        thinned = cv2.ximgproc.thinning(thinned.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

        if report_path is not None:
            cv2.imwrite(str(report_path / "thinned.png"), thinned)

        return thinned, (vessel_classes, vessel_contours)

    def apply(self) -> Tuple[Metadata, Metadata]:

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

        for i in range(img_stack.shape[2]):
            cv2.imwrite(str(report_path / f"slide_{i}.png"), img_stack[:, :, i])

        # Build per-channel views and titles once
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
        else:
            raise ValueError("Unsupported img_stack type for preview.")

        titles = []
        for i, md in enumerate(self.metadata):
            name = getattr(md, "name", None) or getattr(md, "sample_id", None) \
                   or getattr(md, "path", None)
            try:
                name = Path(name).name  # shorten if it's a path
            except Exception:
                pass
            titles.append(str(name) if name is not None else f"image_{i}")

        # Classic preview of all channels
        if self.preview:
            self._preview_images(imgs, titles)

        if self.confirm:
            apply = Confirm.ask("Continue with lobule segmentation?", default=True, console=console)
            if not apply:
                console.print("Aborted by user. No segmentation performed.", style="error")
                return self.metadata

        # Interactive weighting preview
        if self.interactive_weighting:
            self._preview_channel_weighting(img_stack)

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

        thinned, (vessel_classes, vessel_contours) = self.skeletize_kmeans(
            img_stack,
            pad=pad,
            region_size=self.region_size,
            report_path=report_path,
        )

        console.print("Complete. Creating lines segments from skeleton...", style="info")

        # Steps 8: segments
        line_segments = segment_thinned_image(thinned, report_path=report_path)

        console.print("Complete. Creating segmentation mask...", style="info")

        # Step 9: Creating lobule and vessel polygons from line segments and vessel contours
        mask = process_segments_to_mask(
            line_segments,
            thinned.shape,
            cv_contours=[cnt for cnt, class_ in zip(vessel_contours, vessel_classes) if class_ == 0],
            report_path=report_path,
            min_area_px=self.min_area_px,
        )

        # Crop mask by padding
        mask_cropped = mask[pad:-pad, pad:-pad]

        console.print("Complete. Back-mapping mask to all pyramid levels...", style="info")

        # Step 10: Build masks for every level
        mask_pyramid = build_mask_pyramid_from_processed(
            mask_cropped=mask_cropped,
            img_size_base=img_size_base,  # ROI size at base_level (after bbox crop)
            bbox_base=bbox,  # bbox in base_level coords
            orig_shapes=orig_shapes,  # {level: (H,W)} from self.load_image(0)
            base_level=self.base_level,
        )

        console.print("Complete. Creating portality map at every pyramid level...", style="info")

        # Prepare vessel contours in base-level full-frame coordinates
        proc_h, proc_w = mask_cropped.shape
        Hb, Wb = img_size_base
        Hfull_base, Wfull_base = orig_shapes[self.base_level]
        min_r, min_c, max_r, max_c = bbox

        cv_cnt_roi = [c for c, k in zip(vessel_contours, vessel_classes) if k == 0]
        pf_cnt_roi = [c for c, k in zip(vessel_contours, vessel_classes) if k == 1]

        cv_cnt_base = to_base_full(cv_cnt_roi, pad, bbox, (proc_h, proc_w), (Hb, Wb))
        pf_cnt_base = to_base_full(pf_cnt_roi, pad, bbox, (proc_h, proc_w), (Hb, Wb))

        # Compute per-level portality
        portality_pyramid: Dict[int, np.ndarray] = {}
        for lvl, (Hdst, Wdst) in orig_shapes.items():
            full_mask = mask_pyramid[lvl]
            if lvl == self.base_level:
                cv_lvl = cv_cnt_base
                pf_lvl = pf_cnt_base
                P = mask_to_portality(full_mask, cv_lvl, pf_lvl, report_path=None)
            else:
                cv_lvl = rescale_full(cv_cnt_base, Hfull_base, Wfull_base, Hdst, Wdst)
                pf_lvl = rescale_full(pf_cnt_base, Hfull_base, Wfull_base, Hdst, Wdst)
                P = mask_to_portality(full_mask, cv_lvl, pf_lvl, report_path=None)
            portality_pyramid[lvl] = P.astype(np.float32)

        # Crop base-level portality to ROI and save fixed-size PNG
        P_base_full = portality_pyramid[self.base_level]
        portality_cropped = P_base_full[min_r:max_r, min_c:max_c]

        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(alpha=0.0)
        Pm = np.ma.masked_invalid(portality_cropped).astype(np.float32)

        # write ROI-sized portality.png so size matches other images
        plt.imsave(str(report_path / "portality.png"), Pm, cmap=cmap, vmin=0.0, vmax=1.0)

        # Preview after portality is available
        if self.preview:
            orig_vis = img_stack.mean(axis=2).astype(np.uint8)
            overlay_vis = overlay_mask(img_stack, mask_cropped, alpha=0.5)

            # build RGBA from the already-cropped map
            portality_rgba = (cmap(Pm) * 255).astype(np.uint8)
            portality_rgba[..., 3] = (portality_rgba[..., 3] * 0.85).astype(np.uint8)

            self._preview_images(
                [orig_vis, overlay_vis, portality_rgba],
                titles=["Original", "Segmentation Overlay", "Portality"],
            )

        # Save mask pyramid
        seg_path = report_path / f"{new_uid}_seg.tiff"

        new_meta = Metadata(
            path_original=seg_path,
            path_storage=seg_path,
            image_type="mask",
            uid=new_uid+"_mask",
        )
        new_meta.save(report_path)
        save_tif(mask_pyramid, seg_path, metadata=new_meta)

        # Save portality pyramid
        portality_path = report_path / f"{new_uid}_portality.tiff"
        new_port = Metadata(
            path_original=portality_path,
            path_storage=portality_path,
            image_type="portality",
            uid=new_uid+"_portality",
        )
        new_port.save(report_path)
        save_tif(portality_pyramid, portality_path, metadata=new_port)

        console.print("Complete.", style="info")

        # Memory Management
        del (portality_pyramid, P, P_base_full, Pm, full_mask, img_stack, mask, mask_pyramid, mask_cropped, imgs,
             line_segments, cv_cnt_base, cv_cnt_roi, cv_lvl, orig_vis, overlay_vis, pf_cnt_base, pf_cnt_roi, pf_lvl,
             portality_cropped, portality_rgba, thinned, vessel_classes, vessel_contours)

        # Return both metadata objects
        return new_meta, new_port


if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_paths = [DATA_PATH / "reg_n_sep" / "noise.tiff",
                   DATA_PATH / "reg_n_sep" / "periportal.tiff",
                   DATA_PATH / "reg_n_sep" / "perivenous.tiff",
                   ]

    metadata_for_segmentation = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Segmentor = LobuleSegmentor(metadata_for_segmentation,
                                channels_pp=1,
                                channels_pv=2,
                                base_level=0,
                                region_size=25,
                                adaptive_histonorm=True)

    metadata_segmentation, metadata_portality = Segmentor.apply()
