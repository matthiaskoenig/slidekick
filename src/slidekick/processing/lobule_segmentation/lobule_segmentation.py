import numpy as np
from pathlib import Path
from rich.prompt import Confirm
from typing import List, Union, Optional, Tuple, Dict, Any
import cv2
import datetime
import uuid
from sklearn.cluster import KMeans
import napari

import matplotlib.pyplot as plt
from magicgui import magicgui

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
    downsample_stack_to_max_side, holes_from_fg_mask,
    bool_mask_to_uint8, minmax_to_uint8, border_connected_mask,
    render_cluster_gray, nonlinear_channel_weighting, to_base_full, rescale_full,
    common_pyramid_levels, choose_default_preview_level, load_level_from_multiscale,
    discover_pyramid_shapes, preview_images_napari, add_napari_controls_dock,
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
                 bg_low_val: int = 0,
                 clahe_bg_suppress: int = 10,
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
                 vessel_pct_low: float = 5.0,
                 vessel_pct_superpixel_frac: float = 0.7,
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
        @param vessel_zone_ratio_thr_pv: fraction of ring that must be PV
        @param vessel_circularity_min: 4πA/P^2 gate; low keeps elongated vessels
        @param vessel_pct_low: global intensity threshold (0–255) on the per-pixel channel mean used to mark background/holes for vessel detection
        @param vessel_pct_superpixel_frac: minimal fraction of pixels below the intensity threshold inside a superpixel to mark it as background/hole
        @param min_area_px: define area threshold to filter out very small polygons (noise)
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
        self.bg_low_val = int(bg_low_val)
        self.clahe_bg_suppress = int(clahe_bg_suppress)

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
        self.vessel_pct_low = vessel_pct_low
        self.vessel_pct_superpixel_frac = vessel_pct_superpixel_frac

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

        # set missing to 0 (any low-value across channels -> zero all channels)
        # bg_low_val defaults to 0 to preserve previous behavior.
        missing_mask = np.any(image_stack <= int(self.bg_low_val), axis=-1)

        out = np.empty((H, W, N), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

        for c in range(N):
            # copy to avoid mutating the input stack (important for preview caching)
            ch = image_stack[..., c].copy()

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
                ch[ch < int(self.clahe_bg_suppress)] = 0  # suppress background altered by CLAHE

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

    def _interactive_preprocessing_preview(
            self,
            report_path: Optional[Path] = None,
            *,
            require_confirm: bool = False,
    ) -> bool:
        """
        Interactive napari preview for preprocessing (ROI crop + filter) before segmentation.

        Controls:
          - preview_level: pyramid level used for preview computations (common to all stains).
          - bg_low_val: values <= bg_low_val are treated as background.
          - clahe_bg_suppress: after CLAHE, values below this are zeroed.

        Notes:
          - Level choices are the intersection of levels across stains.
          - If levels cannot be determined for a stain, it is treated as [0].
          - Processing runs on a downsampled copy (max side 2048) for responsiveness.

        Returns
        -------
        bool
            True if confirmed (or if require_confirm=False), False if the viewer was closed
            without pressing Confirm.
        """
        from qtpy.QtCore import QTimer

        # Load multiscale handles once; discover common levels from these handles.
        multiscales = [self.load_image(i) for i in range(len(self.metadata))]
        common_levels = common_pyramid_levels(multiscales)
        default_level = choose_default_preview_level(common_levels)

        def _title_for_md(md: Metadata, idx: int) -> str:
            p = getattr(md, "path_original", None) or getattr(md, "path", None) or getattr(md, "name", None)
            if isinstance(p, tuple) and p:
                p = p[0]
            if p is not None:
                try:
                    return Path(p).stem
                except Exception:
                    return str(p)
            return f"image_{idx}"

        titles = [_title_for_md(md, i) for i, md in enumerate(self.metadata)]

        # Cache raw (downsampled) stacks per level; filtered stacks per (level, bg_low_val, clahe_bg_suppress).
        raw_cache: Dict[int, np.ndarray] = {}
        filt_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

        def _get_raw_stack(level: int) -> np.ndarray:
            lvl = int(level)
            if lvl in raw_cache:
                return raw_cache[lvl]

            # Prevent preview from mutating channel lists / throw-out state.
            saved_throw_out = self.throw_out_ratio
            saved_pp = list(self.channels_pp) if self.channels_pp is not None else None
            saved_pv = list(self.channels_pv) if self.channels_pv is not None else None
            try:
                self.throw_out_ratio = None
                stack, _bbox, _orig_shapes = self._load_and_invert_images_from_metadatas(report_path=None, level=lvl)
            finally:
                self.throw_out_ratio = saved_throw_out
                self.channels_pp = saved_pp
                self.channels_pv = saved_pv

            raw_small = downsample_stack_to_max_side(stack, 2048)
            raw_cache[lvl] = raw_small
            return raw_small

        def _get_filtered_stack(level: int, bg_low_val: int, clahe_bg_suppress: int) -> np.ndarray:
            # Always sync instance state to the UI values, even if we hit the cache.
            self.bg_low_val = int(bg_low_val)
            self.clahe_bg_suppress = int(clahe_bg_suppress)

            key = (int(level), int(bg_low_val), int(clahe_bg_suppress))
            if key in filt_cache:
                return filt_cache[key]

            raw = _get_raw_stack(level)
            filt = self._filter(raw)
            filt_cache[key] = filt
            return filt

        viewer = napari.Viewer()

        raw0 = _get_raw_stack(default_level)
        filt0 = _get_filtered_stack(default_level, self.bg_low_val, self.clahe_bg_suppress)

        raw_mean_layer = viewer.add_image(
            raw0.mean(axis=2).astype(np.uint8),
            name="raw/mean",
            colormap="gray",
        )

        raw_layers: List[Any] = []
        filt_layers: List[Any] = []
        for c, title in enumerate(titles):
            raw_layers.append(
                viewer.add_image(
                    raw0[..., c],
                    name=f"raw/{title}",
                    colormap="gray",
                    visible=False,
                )
            )
            filt_layers.append(
                viewer.add_image(
                    filt0[..., c],
                    name=f"filtered/{title}",
                    colormap="gray",
                )
            )

        pending = {
            "level": int(default_level),
            "bg": int(self.bg_low_val),
            "clahe": int(self.clahe_bg_suppress),
        }

        confirmed = {"ok": False}

        timer = QTimer()
        timer.setSingleShot(True)

        def _apply_update() -> None:
            lvl = int(pending["level"])
            bg = int(pending["bg"])
            clahe = int(pending["clahe"])

            raw = _get_raw_stack(lvl)
            filt = _get_filtered_stack(lvl, bg, clahe)

            raw_mean_layer.data = raw.mean(axis=2).astype(np.uint8)
            for c in range(raw.shape[2]):
                raw_layers[c].data = raw[..., c]
                filt_layers[c].data = filt[..., c]

        timer.timeout.connect(_apply_update)

        @magicgui(
            layout="vertical",
            auto_call=True,
            preview_level={"choices": common_levels},
            bg_low_val={"min": 0, "max": 255, "step": 1},
            clahe_bg_suppress={"min": 0, "max": 255, "step": 1},
        )
        def controls(
                preview_level: int = int(default_level),
                bg_low_val: int = int(self.bg_low_val),
                clahe_bg_suppress: int = int(self.clahe_bg_suppress),
        ):
            pending["level"] = int(preview_level)
            pending["bg"] = int(bg_low_val)
            pending["clahe"] = int(clahe_bg_suppress)

            # Debounce: restart timer on every change.
            timer.start(200)

        def on_update() -> None:
            timer.stop()
            _apply_update()

        def on_confirm() -> None:
            timer.stop()
            _apply_update()
            confirmed["ok"] = True
            viewer.close()

        add_napari_controls_dock(
            viewer,
            controls,
            on_update=on_update,
            on_confirm=on_confirm,
            include_update=True,
            include_confirm=True,
            include_reset=False,
            update_text="Preview / Update",
            confirm_text="Confirm",
        )

        napari.run()

        return True if not require_confirm else bool(confirmed["ok"])

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
        - Updates overlays on a short debounce timer and provides a
          'Preview / Update' button to force recomputation immediately.
        - Provides 'Reset to defaults' and 'Confirm and continue' buttons.
        """
        from qtpy.QtCore import QTimer

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

        # Downsample whole stack for preview (fast).
        stack_small = downsample_stack_to_max_side(image_stack, 2048)

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

        pending = dict(defaults)

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

            channels_pp = [c for c in (self.channels_pp or []) if 0 <= c < C]
            channels_pv = [c for c in (self.channels_pv or []) if 0 <= c < C]

            if self.nonlinear_kmeans:
                X = nonlinear_channel_weighting(
                    flat,
                    channels_pp,
                    channels_pv,
                    self.pp_gamma,
                    self.pv_gamma,
                    self.nl_low_pct,
                    self.nl_high_pct,
                )
            else:
                X = flat

            w_feat = np.ones(X.shape[1], dtype=np.float32)
            for idx_ch in channels_pp:
                w_feat[idx_ch] *= float(self.alpha_pp)
            for idx_ch in channels_pv:
                w_feat[idx_ch] *= float(self.alpha_pv)

            Xw_img = (X * w_feat).reshape(H, W, C)

            pp_img: Optional[np.ndarray] = None
            pv_img: Optional[np.ndarray] = None

            if channels_pp:
                pp_vals = Xw_img[..., channels_pp]
                pp_float = pp_vals if pp_vals.ndim == 2 else pp_vals.mean(axis=2)
                pp_img = minmax_to_uint8(pp_float)

            if channels_pv:
                pv_vals = Xw_img[..., channels_pv]
                pv_float = pv_vals if pv_vals.ndim == 2 else pv_vals.mean(axis=2)
                pv_img = minmax_to_uint8(pv_float)

            return pp_img, pv_img

        # Napari viewer
        viewer = napari.Viewer()
        # TODO: Integrate with slidekick viewer

        pp_layer = None
        pv_layer = None

        def _apply_update() -> None:
            """
            Push pending parameters into self, recompute composites, update layers.
            """
            self.nonlinear_kmeans = bool(pending["nonlinear_kmeans"])
            self.alpha_pp = float(pending["alpha_pp"])
            self.alpha_pv = float(pending["alpha_pv"])
            self.pp_gamma = float(pending["pp_gamma"])
            self.pv_gamma = float(pending["pv_gamma"])
            self.nl_low_pct = float(pending["nl_low_pct"])
            self.nl_high_pct = float(pending["nl_high_pct"])

            new_pp, new_pv = compute_pp_pv_images()

            nonlocal pp_layer, pv_layer
            if new_pp is not None:
                if pp_layer is None:
                    pp_layer = viewer.add_image(
                        new_pp,
                        name="PP channel (weighted)",
                        colormap="green",
                        blending="additive",
                        opacity=0.5,
                    )
                else:
                    pp_layer.data = new_pp

            if new_pv is not None:
                if pv_layer is None:
                    pv_layer = viewer.add_image(
                        new_pv,
                        name="PV channel (weighted)",
                        colormap="magenta",
                        blending="additive",
                        opacity=0.5,
                    )
                else:
                    pv_layer.data = new_pv

        # Initial render
        pending.update(
            nonlinear_kmeans=self.nonlinear_kmeans,
            alpha_pp=self.alpha_pp,
            alpha_pv=self.alpha_pv,
            pp_gamma=self.pp_gamma,
            pv_gamma=self.pv_gamma,
            nl_low_pct=self.nl_low_pct,
            nl_high_pct=self.nl_high_pct,
        )
        _apply_update()

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(_apply_update)

        @magicgui(
            layout="vertical",
            auto_call=True,  # stage updates on every change; apply on debounce or button
            nonlinear_kmeans={"widget_type": "CheckBox"},
            alpha_pp={"min": 0.0, "max": 5.0, "step": 0.01},
            alpha_pv={"min": 0.0, "max": 5.0, "step": 0.01},
            pp_gamma={"min": 0.1, "max": 5.0, "step": 0.01},
            pv_gamma={"min": 0.1, "max": 5.0, "step": 0.01},
            nl_low_pct={"min": 0.0, "max": 50.0, "step": 0.5},
            nl_high_pct={"min": 50.0, "max": 100.0, "step": 0.5},
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
            Stage parameter changes; actual recomputation happens on a short debounce
            timer or when the user presses Preview / Update.
            """
            pending["nonlinear_kmeans"] = bool(nonlinear_kmeans)
            pending["alpha_pp"] = float(alpha_pp)
            pending["alpha_pv"] = float(alpha_pv)
            pending["pp_gamma"] = float(pp_gamma)
            pending["pv_gamma"] = float(pv_gamma)
            pending["nl_low_pct"] = float(nl_low_pct)
            pending["nl_high_pct"] = float(nl_high_pct)

            timer.start(200)

        def on_update() -> None:
            timer.stop()
            _apply_update()

        def on_reset() -> None:
            timer.stop()

            # restore defaults in pending and in GUI widgets
            pending.update(defaults)

            weighting_controls.nonlinear_kmeans.value = defaults["nonlinear_kmeans"]
            weighting_controls.alpha_pp.value = defaults["alpha_pp"]
            weighting_controls.alpha_pv.value = defaults["alpha_pv"]
            weighting_controls.pp_gamma.value = defaults["pp_gamma"]
            weighting_controls.pv_gamma.value = defaults["pv_gamma"]
            weighting_controls.nl_low_pct.value = defaults["nl_low_pct"]
            weighting_controls.nl_high_pct.value = defaults["nl_high_pct"]

            _apply_update()

        def on_confirm() -> None:
            timer.stop()
            _apply_update()
            viewer.close()

        add_napari_controls_dock(
            viewer,
            weighting_controls,
            on_update=on_update,
            on_confirm=on_confirm,
            on_reset=on_reset,
            include_update=True,
            include_confirm=True,
            include_reset=True,
            update_text="Preview / Update",
            confirm_text="Confirm and continue",
            reset_text="Reset to defaults",
        )

        napari.run()


    def _load_and_invert_images_from_metadatas(
            self,
            report_path: Optional[Path] = None,
            level: Optional[int] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Dict[int, Any]]:
        """
        Load each image at the pyramid level defined, invert selected channels,
        harmonize shapes, and stack along the last axis -> (H, W, N).
        """

        lvl = int(self.base_level if level is None else level)

        arrays: List[np.ndarray] = []
        for i, _md in enumerate(self.metadata):
            try:
                img = load_level_from_multiscale(self.load_image(i), lvl)
            except Exception as e:
                raise Exception(f"Level {lvl} failed for image {i} in stack") from e

            img = np.asarray(img)
            # ensure 2D
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = np.mean(img[..., :3], axis=-1).astype(img.dtype)

            arrays.append(img)

        if not arrays:
            raise RuntimeError("No images loaded after filtering.")

        # harmonize shapes before stacking
        h_min = min(a.shape[0] for a in arrays)
        w_min = min(a.shape[1] for a in arrays)
        arrays = [a[:h_min, :w_min] for a in arrays]

        stack = np.dstack(arrays).astype(np.uint8)

        # Get resolutions / image sizes for later back mapping based on stain 0
        orig_shapes = discover_pyramid_shapes(self.load_image(0))

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
        # NOTE: Discarding channels changes channel indices. We first decide what to drop,
        # then delete in reverse order to avoid index shifting bugs.
        drop: List[int] = []

        for i in range(stack.shape[-1]):
            # optional inversion
            if self.channels is not None and i in self.channels:
                if np.issubdtype(stack[:, :, i].dtype, np.integer):
                    info = np.iinfo(stack[:, :, i].dtype)
                    stack[:, :, i] = info.max - stack[:, :, i]
                else:  # float images assumed in [0,1]
                    stack[:, :, i] = 1.0 - stack[:, :, i]

            if self.throw_out_ratio is not None:
                non_bg = np.count_nonzero(stack[:, :, i] != 255)
                ratio = non_bg / float(stack[:, :, i].size)
                if ratio < float(self.throw_out_ratio):
                    drop.append(i)

        if drop:
            for i in sorted(drop, reverse=True):
                console.print(
                    f"Discarded {i} with non-background pixel ratio: "
                    f"{np.count_nonzero(stack[:, :, i] != 255) / float(stack[:, :, i].size):.3f}",
                    style="warning",
                )
                stack = np.delete(stack, i, axis=-1)

                # Keep PP/PV channel lists consistent with the new stack.
                if self.channels_pp is not None:
                    if i in self.channels_pp:
                        console.print("Channel used for periportal detection removed.", style="warning")
                    self.channels_pp = [j - 1 if j > i else j for j in self.channels_pp if j != i]

                if self.channels_pv is not None:
                    if i in self.channels_pv:
                        console.print("Channel used for perivenous detection removed.", style="warning")
                    self.channels_pv = [j - 1 if j > i else j for j in self.channels_pv if j != i]

            if (self.channels_pp is not None and not self.channels_pp) and (
                    self.channels_pv is not None and not self.channels_pv):
                console.print("No remaining channels for perivenous or periportal detection available.", style="error")
                raise Exception("No remaining channels for segmentation available.")

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

        base_fg_label_mask = ~(dark_frac > 0.5).any(axis=1)

        # Keep base_gray for display only
        base_gray = image_stack.mean(axis=2).astype(np.uint8)

        # IMPORTANT: use float per-pixel mean (old behavior), not uint8-cast mean
        # This matters when thresholds are very low (e.g. 3–10).
        gray_flat = img_flat.mean(axis=1).astype(np.float32)

        # Determine pixels that are definitely outside ROI (connected to border and ==0 in all channels)
        # Pixels that are exactly 0 across all channels correspond to padded/outside ROI area.
        # Exclude the connected components of that region that touch the image border.
        border_bg_px = border_connected_mask(image_stack.sum(axis=2) == 0, connectivity=8)

        # Convert border-bg pixels to border-bg SUPERPIXELS once.
        border_bg_sp_ids = None
        if border_bg_px is not None:
            try:
                border_bg_sp_ids = np.unique(labels[border_bg_px])
            except Exception:
                border_bg_sp_ids = None

        def compute_fg_masks(
                vessel_pct_low_val: float,
                vessel_pct_superpixel_frac_val: float,
        ):
            """
            Old, working behavior:
              - thr_gray is an ABSOLUTE intensity threshold on per-pixel mean (0..255).
              - A superpixel becomes a vessel/hole candidate if
                    frac(pixels with mean <= thr_gray) >= vessel_pct_superpixel_frac_val
              - Candidate superpixels are then PUNCHED OUT of FG so they become holes.
            """
            fg_mask_local = base_fg_label_mask.copy()

            # Ensure border-connected outside region cannot accidentally become FG.
            if border_bg_sp_ids is not None and getattr(border_bg_sp_ids, "size", 0) > 0:
                fg_mask_local[border_bg_sp_ids] = False

            vessel_labels = np.zeros_like(base_fg_label_mask, dtype=bool)

            thr_gray = float(vessel_pct_low_val)
            if thr_gray > 0.0:
                gray_dark = (gray_flat <= thr_gray).astype(np.float32)
                gray_dark_counts = np.bincount(lab_flat, weights=gray_dark, minlength=num_labels)
                gray_dark_frac = np.divide(gray_dark_counts, counts, where=nz)

                # Candidate vessel/hole superpixels (restricted to FG label set)
                vessel_labels = (gray_dark_frac >= float(vessel_pct_superpixel_frac_val)) & fg_mask_local

            # Optional pruning of tiny candidates (same as your new code, but applied to vessel_labels)
            min_area_sp = int(min(self.min_vessel_area_pp, self.min_vessel_area_pv))
            if min_area_sp > 0:
                candidate_label_ids = np.nonzero(vessel_labels)[0]
                if candidate_label_ids.size > 0:
                    vessel_mask_px = np.isin(labels, candidate_label_ids)
                    num_cc, cc_labels, stats, _centroids = cv2.connectedComponentsWithStats(
                        vessel_mask_px.astype(np.uint8),
                        connectivity=8,
                    )
                    for cc_id in range(1, num_cc):
                        area_cc = int(stats[cc_id, cv2.CC_STAT_AREA])
                        if area_cc < min_area_sp:
                            cc_mask = (cc_labels == cc_id)
                            sp_ids = np.unique(labels[cc_mask])
                            vessel_labels[sp_ids] = False

            # CRITICAL: punch candidates out of FG so they become holes for contour-based vessel detection
            fg_mask_local = fg_mask_local & (~vessel_labels)

            # Recompute FG pixels after punch-out
            fg_labels_local = np.nonzero(fg_mask_local)[0]
            fg_pix_mask_local = np.isin(labels, fg_labels_local)

            # Candidate pixels (for debugging/preview layers)
            candidate_label_ids = np.nonzero(vessel_labels)[0]
            vessel_pix_mask_local = np.isin(labels, candidate_label_ids)

            return fg_mask_local, fg_labels_local, fg_pix_mask_local, vessel_pix_mask_local

        if self.interactive_vessels:

            stored_vals = [
                self.vessel_pct_low,
                self.vessel_pct_superpixel_frac,
                int(min(self.min_vessel_area_pp, self.min_vessel_area_pv)),
            ]

            fg_label_mask, fg_labels, fg_pix_mask, vessel_pix_mask = compute_fg_masks(
                self.vessel_pct_low,
                self.vessel_pct_superpixel_frac,
            )

            base_gray = image_stack.mean(axis=2).astype(np.uint8)
            viewer = napari.Viewer()
            base_layer = viewer.add_image(
                base_gray,
                name="base_gray",
                colormap="gray",
                blending="translucent",
            )

            hole_preview = holes_from_fg_mask(fg_pix_mask, border_exclude=border_bg_px)

            vessel_layer = viewer.add_image(
                bool_mask_to_uint8(hole_preview),
                name="vessel candidates (holes)",
                colormap="cyan",
                opacity=1.0,
                blending="additive",
            )

            # Optional: keep the original candidate superpixels as a second debug layer
            vessel_sp_layer = viewer.add_image(
                bool_mask_to_uint8(vessel_pix_mask),
                name="vessel candidates (superpixels)",
                colormap="yellow",
                opacity=0.7,
                blending="additive",
                visible=False,
            )

            pending = {
                "vessel_pct_low": float(self.vessel_pct_low),
                "vessel_pct_superpixel_frac": float(self.vessel_pct_superpixel_frac),
                "min_vessel_area": int(min(self.min_vessel_area_pp, self.min_vessel_area_pv)),
            }

            @magicgui(
                layout="vertical",
                auto_call=True,  # stage values on change
                vessel_pct_low={"min": 0.0, "max": 20.0, "step": 0.5},
                vessel_pct_superpixel_frac={"min": 0.0, "max": 1.0, "step": 0.01},
                min_vessel_area={"min": 0, "max": 20000, "step": 50},
            )
            def vessel_controls(
                    vessel_pct_low: float = self.vessel_pct_low,
                    vessel_pct_superpixel_frac: float = self.vessel_pct_superpixel_frac,
                    min_vessel_area: int = int(min(self.min_vessel_area_pp, self.min_vessel_area_pv)),
            ):
                pending["vessel_pct_low"] = float(vessel_pct_low)
                pending["vessel_pct_superpixel_frac"] = float(vessel_pct_superpixel_frac)
                pending["min_vessel_area"] = int(min_vessel_area)

            def _apply_vessel_candidate_update() -> None:
                self.vessel_pct_low = float(pending["vessel_pct_low"])
                self.vessel_pct_superpixel_frac = float(pending["vessel_pct_superpixel_frac"])
                min_area_int = int(pending["min_vessel_area"])
                self.min_vessel_area_pp = min_area_int
                self.min_vessel_area_pv = min_area_int

                _, _, fg_pix_mask_local, vessel_pix_mask_local = compute_fg_masks(
                    self.vessel_pct_low,
                    self.vessel_pct_superpixel_frac,
                )

                hole_preview_local = holes_from_fg_mask(fg_pix_mask_local)
                vessel_layer.data = (hole_preview_local.astype(np.uint8) * 255)

                # Optional debug layer update
                vessel_sp_layer.data = (vessel_pix_mask_local.astype(np.uint8) * 255)

            def on_reset() -> None:
                self.vessel_pct_low = stored_vals[0]
                self.vessel_pct_superpixel_frac = stored_vals[1]
                min_area_int = int(stored_vals[2])
                self.min_vessel_area_pp = min_area_int
                self.min_vessel_area_pv = min_area_int

                pending["vessel_pct_low"] = float(self.vessel_pct_low)
                pending["vessel_pct_superpixel_frac"] = float(self.vessel_pct_superpixel_frac)
                pending["min_vessel_area"] = int(min_area_int)

                vessel_controls.vessel_pct_low.value = float(self.vessel_pct_low)
                vessel_controls.vessel_pct_superpixel_frac.value = float(self.vessel_pct_superpixel_frac)
                vessel_controls.min_vessel_area.value = int(min_area_int)

                _apply_vessel_candidate_update()

            def on_confirm() -> None:
                _apply_vessel_candidate_update()
                viewer.close()

            add_napari_controls_dock(
                viewer,
                vessel_controls,
                on_update=_apply_vessel_candidate_update,
                on_confirm=on_confirm,
                on_reset=on_reset,
                include_update=True,
                include_confirm=True,
                include_reset=True,
                update_text="Preview / Update",
                confirm_text="Confirm and continue",
                reset_text="Reset vessel detection params",
            )

            napari.run()

            fg_label_mask, fg_labels, fg_pix_mask, vessel_pix_mask = compute_fg_masks(
                self.vessel_pct_low,
                self.vessel_pct_superpixel_frac,
            )
        else:
            fg_label_mask, fg_labels, fg_pix_mask, vessel_pix_mask = compute_fg_masks(
                self.vessel_pct_low,
                self.vessel_pct_superpixel_frac,
            )

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

            # Candidate outputs:
            # 1) superpixel-based candidates (threshold stage)
            cv2.imwrite(
                str(report_path / "vessel_candidates_superpixels.png"),
                bool_mask_to_uint8(vessel_pix_mask),
            )

            # 2) hole-based candidates (topology stage; matches vessel contour candidate concept)
            hole_candidates = holes_from_fg_mask(fg_pix_mask, border_exclude=border_bg_px)

            cv2.imwrite(
                str(report_path / "vessel_candidates_holes.png"),
                bool_mask_to_uint8(hole_candidates),
            )

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

        X = X_means.copy()
        C = X.shape[1]

        if self.nonlinear_kmeans:
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
        # Faster than np.vectorize: build a LUT once and index it
        lut = np.full((num_labels,), -1, dtype=np.int32)
        for sp_id, cid in assigned_by_sp.items():
            sid = int(sp_id)
            if 0 <= sid < num_labels:
                lut[sid] = int(cid)

        cluster_map = lut[labels].astype(np.int32)  # shape (H, W), values in {-1, 0..n_clusters-1}

        if report_path is not None:
            template_init = render_cluster_gray(cluster_map, sorted_label_idx, self.n_clusters)
            template_init[~fg_pix_mask] = 0
            cv2.imwrite(str(report_path / "foreground_clustered.png"), template_init)

        console.print("Complete. Now detecting vessels in detected zonation...", style="info")

        # keep a copy of the pure KMeans assignment so we can restart from it
        assigned_by_sp_kmeans: Dict[int, int] = dict(assigned_by_sp)

        # approximate tissue mask from current stack
        base_gray = image_stack.mean(axis=2).astype(np.uint8)
        tissue_bool = base_gray > 0
        bg_bool = ~tissue_bool

        def run_vessel_detection(
                min_vessel_area_pp: int,
                min_vessel_area_pv: int,
                vessel_annulus_px: int,
                vessel_zone_ratio_thr_pp: float,
                vessel_zone_ratio_thr_pv: float,
                vessel_circularity_min: float,
        ) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
            """
            Run vessel detection and superpixel reassignment on top of the
            fixed SLIC + KMeans result.

            This uses the current KMeans based cluster_map, labels, fg_pix_mask,
            idx_pp / idx_pv / idx_mid etc, but only changes the vessel related
            parameters.

            Returns
            -------
            cluster_map_final : (H, W) int32
                Final zonation map after vessel based reassignment.
            vessel_classes : list of int
                One entry per contour, 0 for PV, 1 for PP.
            vessel_contours : list of np.ndarray
                Contours corresponding to detected vessels.
            """
            H, W, _ = image_stack.shape

            # start each run from the unmodified KMeans assignment
            assigned_by_sp_local: Dict[int, int] = dict(assigned_by_sp_kmeans)

            # Fast LUT mapping is built once at the end; avoid np.vectorize here.

            # 4) Vessel detection and superpixel reassignment
            vessel_classes: List[int] = []
            vessel_contours: List[np.ndarray] = []

            k = max(1, int(vessel_annulus_px))
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

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
                min_area = min(int(min_vessel_area_pp), int(min_vessel_area_pv))
                circ_min = float(vessel_circularity_min) * 0.9

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

            mask_all = np.zeros((H, W), dtype=np.uint8)
            # Use full tissue support. Holes (cluster_map < 0) become child contours.
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

                pp_pass = pp_frac >= float(vessel_zone_ratio_thr_pp)
                pv_pass = pv_frac >= float(vessel_zone_ratio_thr_pv)
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
                    assigned_by_sp_local[int(sp)] = zone_idx
                vessel_contours.append(cnt)
                vessel_classes.append(cls)

            # Rebuild final per-pixel cluster map after reassignment (fast LUT)
            lut_local = np.full((num_labels,), -1, dtype=np.int32)
            for sp_id, cid in assigned_by_sp_local.items():
                sid = int(sp_id)
                if 0 <= sid < num_labels:
                    lut_local[sid] = int(cid)
            cluster_map_final = lut_local[labels].astype(np.int32)

            return cluster_map_final, vessel_classes, vessel_contours

        # remember current vessel params as defaults for this run
        defaults = dict(
            min_vessel_area_pp=self.min_vessel_area_pp,
            min_vessel_area_pv=self.min_vessel_area_pv,
            vessel_annulus_px=self.vessel_annulus_px,
            vessel_zone_ratio_thr_pp=self.vessel_zone_ratio_thr_pp,
            vessel_zone_ratio_thr_pv=self.vessel_zone_ratio_thr_pv,
            vessel_circularity_min=self.vessel_circularity_min,
        )

        if self.interactive_vessels:
            # initial run with current parameters (reuses SLIC + KMeans result)
            cluster_map_prev, vessel_classes_prev, vessel_contours_prev = run_vessel_detection(
                self.min_vessel_area_pp,
                self.min_vessel_area_pv,
                self.vessel_annulus_px,
                self.vessel_zone_ratio_thr_pp,
                self.vessel_zone_ratio_thr_pv,
                self.vessel_circularity_min,
            )

            # base image and helper for overlay
            base_vis = image_stack.mean(axis=2).astype(np.uint8)
            base_rgb = cv2.cvtColor(base_vis, cv2.COLOR_GRAY2BGR)

            COLOR_PP = (255, 0, 255)
            COLOR_MID = (60, 160, 60)
            COLOR_PV = (0, 165, 255)

            def make_overlay(cm_local: np.ndarray,
                             vclasses_local: List[int],
                             vcontours_local: List[np.ndarray]) -> np.ndarray:
                """""Build zonation + vessel overlay as RGB image."""
                zon_rgb = np.zeros_like(base_rgb, dtype=np.uint8)

                mask_pp = (cm_local == idx_pp) & fg_pix_mask
                mask_pv = (cm_local == idx_pv) & fg_pix_mask
                mask_mid = np.zeros_like(mask_pp, dtype=bool)
                for mid_idx in idx_mid:
                    mask_mid |= (cm_local == mid_idx)
                mask_mid &= fg_pix_mask

                zon_rgb[mask_pp] = COLOR_PP
                zon_rgb[mask_mid] = COLOR_MID
                zon_rgb[mask_pv] = COLOR_PV

                ov_bgr = cv2.addWeighted(base_rgb, 0.6, zon_rgb, 0.4, 0.0)
                for cnt, cls in zip(vcontours_local, vclasses_local):
                    # central (PV) = yellow, portal (PP) = magenta
                    color = (255, 255, 0) if cls == 0 else (255, 0, 255)
                    cv2.drawContours(ov_bgr, [cnt], -1, color, thickness=2)

                return cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)

            overlay_rgb = make_overlay(cluster_map_prev, vessel_classes_prev, vessel_contours_prev)

            viewer = napari.Viewer()
            overlay_layer = viewer.add_image(
                overlay_rgb,
                name="Zonation + vessels (preview)",
                rgb=True,
                blending="translucent",
            )

            # sliders + a single "Change" button created by magicgui
            @magicgui(
                layout="vertical",
                auto_call=False,
                call_button="Change (recalculate vessels)",  # rename magicgui's Run button
                min_vessel_area_pp={"min": 0, "max": 10000, "step": 50},
                min_vessel_area_pv={"min": 0, "max": 10000, "step": 50},
                vessel_annulus_px={"min": 1, "max": 100, "step": 1},
                vessel_zone_ratio_thr_pp={"min": 0.0, "max": 1.0, "step": 0.01},
                vessel_zone_ratio_thr_pv={"min": 0.0, "max": 1.0, "step": 0.01},
                vessel_circularity_min={"min": 0.0, "max": 1.0, "step": 0.01},
            )
            def vessel_controls(
                    min_vessel_area_pp: int = self.min_vessel_area_pp,
                    min_vessel_area_pv: int = self.min_vessel_area_pv,
                    vessel_annulus_px: int = self.vessel_annulus_px,
                    vessel_zone_ratio_thr_pp: float = self.vessel_zone_ratio_thr_pp,
                    vessel_zone_ratio_thr_pv: float = self.vessel_zone_ratio_thr_pv,
                    vessel_circularity_min: float = self.vessel_circularity_min,
            ):
                # push values from GUI into object state
                self.min_vessel_area_pp = int(min_vessel_area_pp)
                self.min_vessel_area_pv = int(min_vessel_area_pv)
                self.vessel_annulus_px = int(vessel_annulus_px)
                self.vessel_zone_ratio_thr_pp = float(vessel_zone_ratio_thr_pp)
                self.vessel_zone_ratio_thr_pv = float(vessel_zone_ratio_thr_pv)
                self.vessel_circularity_min = float(vessel_circularity_min)

                console.print(
                    "Updated vessel gating params: "
                    f"min_pp={self.min_vessel_area_pp}, "
                    f"min_pv={self.min_vessel_area_pv}, "
                    f"annulus={self.vessel_annulus_px}, "
                    f"zone_thr_pp={self.vessel_zone_ratio_thr_pp:.2f}, "
                    f"zone_thr_pv={self.vessel_zone_ratio_thr_pv:.2f}, "
                    f"circ_min={self.vessel_circularity_min:.2f}",
                    style="info",
                )

                # recompute vessels only
                cm_local, vclasses_local, vcontours_local = run_vessel_detection(
                    self.min_vessel_area_pp,
                    self.min_vessel_area_pv,
                    self.vessel_annulus_px,
                    self.vessel_zone_ratio_thr_pp,
                    self.vessel_zone_ratio_thr_pv,
                    self.vessel_circularity_min,
                )
                overlay_layer.data = make_overlay(cm_local, vclasses_local, vcontours_local)

            def on_reset() -> None:
                # restore defaults in self
                self.min_vessel_area_pp = defaults["min_vessel_area_pp"]
                self.min_vessel_area_pv = defaults["min_vessel_area_pv"]
                self.vessel_annulus_px = defaults["vessel_annulus_px"]
                self.vessel_zone_ratio_thr_pp = defaults["vessel_zone_ratio_thr_pp"]
                self.vessel_zone_ratio_thr_pv = defaults["vessel_zone_ratio_thr_pv"]
                self.vessel_circularity_min = defaults["vessel_circularity_min"]

                # restore GUI values
                vessel_controls.min_vessel_area_pp.value = defaults["min_vessel_area_pp"]
                vessel_controls.min_vessel_area_pv.value = defaults["min_vessel_area_pv"]
                vessel_controls.vessel_annulus_px.value = defaults["vessel_annulus_px"]
                vessel_controls.vessel_zone_ratio_thr_pp.value = defaults["vessel_zone_ratio_thr_pp"]
                vessel_controls.vessel_zone_ratio_thr_pv.value = defaults["vessel_zone_ratio_thr_pv"]
                vessel_controls.vessel_circularity_min.value = defaults["vessel_circularity_min"]

                # re-run with defaults (same as pressing Change)
                vessel_controls()

            def on_confirm() -> None:
                viewer.close()

            # Keep the "Change (recalculate vessels)" button inside `vessel_controls`
            # (auto_call=False), and add only Confirm/Reset here.
            add_napari_controls_dock(
                viewer,
                vessel_controls,
                on_confirm=on_confirm,
                on_reset=on_reset,
                include_update=False,
                include_confirm=True,
                include_reset=True,
                confirm_text="Confirm and continue",
                reset_text="Reset vessel params",
            )

            napari.run()

        # final vessel detection with whatever parameters the user ended up with
        cluster_map_final, vessel_classes, vessel_contours = run_vessel_detection(
            self.min_vessel_area_pp,
            self.min_vessel_area_pv,
            self.vessel_annulus_px,
            self.vessel_zone_ratio_thr_pp,
            self.vessel_zone_ratio_thr_pv,
            self.vessel_circularity_min,
        )

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

        # Initial interactive preprocessing preview (napari):
        # - preview_level selector uses intersection across stains
        # - bg_low_val and clahe_bg_suppress update immediately (debounced)
        # - confirm is handled in the napari UI when preview is enabled
        if self.preview:
            ok = self._interactive_preprocessing_preview(report_path=report_path, require_confirm=self.confirm)
            if self.confirm and not ok:
                console.print("Aborted by user. No segmentation performed.", style="error")
                return self.metadata

        console.print("Loading images...", style="info")
        img_stack, bbox, orig_shapes = self._load_and_invert_images_from_metadatas(report_path)

        img_size_base = img_stack.shape[:2]  # base size for back-mapping of mask

        console.print("Complete. Now applying filters...", style="info")
        img_stack = self._filter(img_stack)

        for i in range(img_stack.shape[2]):
            cv2.imwrite(str(report_path / f"slide_{i}.png"), img_stack[:, :, i])

        if (not self.preview) and self.confirm:
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

            preview_images_napari(
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
        del (portality_pyramid, P, P_base_full, Pm, full_mask, img_stack, mask, mask_pyramid, mask_cropped,
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
