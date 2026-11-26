from typing import List, Optional
import datetime
import uuid
from pathlib import Path
import shutil
from rich.prompt import Confirm
import warnings
import os

from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH
from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.czi import czi2tiff

# VALIS imports
from valis import registration, slide_io
from valis import feature_detectors, feature_matcher, non_rigid_registrars, warp_tools
from skimage import transform as sktransform

import numpy as np
import pyvips

# ---------------------------------------------------------------------------
# Backend choices exposed to the user. These select the actual VALIS algorithm
# used for registration (as opposed to just the preview/stage shown).
# ---------------------------------------------------------------------------
RIGID_TRANSFORMS = {
    "similarity": sktransform.SimilarityTransform,  # rotation + translation + uniform scale (default)
    "euclidean": sktransform.EuclideanTransform,    # rigid body: rotation + translation only
    "affine": sktransform.AffineTransform,          # + shear / anisotropic scale
}

# Non-rigid backends. "none" disables non-rigid refinement (rigid-only result).
NON_RIGID_METHODS = {
    "optical_flow": non_rigid_registrars.OpticalFlowWarper,  # default
    "elastix": non_rigid_registrars.SimpleElastixWarper,     # B-spline (SimpleElastix)
    "raft": non_rigid_registrars.RAFTWarper,                 # deep optical flow (RAFT)
    "none": None,
}

# Feature detectors. "default" keeps VALIS' modern LightGlue + Disk deep-matching
# pipeline. Any other choice switches to classic detect-and-match, and we pair the
# detector with a matching classic Matcher so the two stay consistent.
FEATURE_DETECTORS = {
    "default": None,
    "vgg": feature_detectors.VggFD,
    "orb": feature_detectors.OrbFD,
    "brisk": feature_detectors.BriskFD,
    "akaze": feature_detectors.AkazeFD,
    "sensitive": feature_detectors.SuperPointFD,
}

class ValisRegistrator(BaseOperator):
    """
    Registers slides using VALIS. For now: Only highest resolution level is supported.
    """

    def __init__(
        self,
        metadata: List[Metadata],
        save_img: bool = True,
        imgs_ordered: bool = False,
        max_processed_image_dim_px: int = 850,
        max_non_rigid_registration_dim_px: int = 850,
        max_micro_registration_dim_px: int = 1700,
        error_threshold_px: float = 5.0,
        max_attempts: int = 3,
        upscale_factor: float = 1.5,
        confirm: bool = True,
        preview: bool = True,
        adaptive_loop: bool = False,
        micro_registration: bool = False,
        crop: str = "reference",
        transform_czi: Optional[List[int]] = None,
        rigid_transform: str = "similarity",
        non_rigid_method: str = "optical_flow",
        feature_detector: str = "default",
        registration_stage: str = "auto",
        preview_max_dim_px: int = 600,
    ):

        """
        channel_selection is ignored for registration (we register whole images),
        but we keep the argument signature compatible with BaseOperator usage.

        Parameters
        ----------
        metadata
            List of Metadata objects to register.
        max_processed_image_dim_px
            Starting processed-image max dimension in pixels.
        max_non_rigid_registration_dim_px
            Starting non-rigid registration max dimension in pixels.
        max_micro_registration_dim_px
            Micro registration max dimension in pixels.
        error_threshold_px
            Target median feature-distance error in pixels. If the measured
            error is larger, the registration will be retried with upsampled
            processed / non-rigid dimensions.
        max_attempts
            Maximum number of registration attempts with increasing resolution.
        upscale_factor
            Multiplicative factor to increase processed / non-rigid dimensions
            between attempts.
        adaptive_loop
            Adaptively upsample based on error metric
        micro_registration
            Use micro registration
        crop
            Cropping style of VALIS
        rigid_transform
            Rigid transform model: "similarity" (default), "euclidean"
            (rigid body, no scale), or "affine".
        non_rigid_method
            Non-rigid backend: "optical_flow" (default), "elastix", "raft",
            or "none" to skip non-rigid refinement (rigid-only result).
        feature_detector
            Feature backend: "default" (VALIS' LightGlue + Disk deep matching),
            or a classic detector "vgg" / "orb" / "brisk" / "akaze" /
            "sensitive" (SuperPoint).
        registration_stage
            Which stage's transform to apply to the full-resolution slides:
            "rigid", "non_rigid", "micro", or "auto" (best available). With an
            interactive napari preview this is the pre-selected default; the
            final choice is whatever stage is active when you click Apply.
        preview_max_dim_px
            Long-side resolution (px) at which warped stages are rendered in the
            napari preview. The warp reads the smallest pyramid level >= this
            size, so cost is bounded regardless of slide size. Higher = more
            detail to judge micro alignment, but slower to build. Default 600;
            the napari preview seeds its "preview px" field from this value.
        """
        self._validate_backend(rigid_transform, non_rigid_method, feature_detector, registration_stage)
        self.rigid_transform = str(rigid_transform)
        self.non_rigid_method = str(non_rigid_method)
        self.feature_detector = str(feature_detector)
        self.registration_stage = str(registration_stage)
        self.preview_max_dim_px = int(preview_max_dim_px)
        self.save_img = save_img
        channel_selection = None
        self.imgs_ordered = imgs_ordered
        self.max_processed_image_dim_px = max_processed_image_dim_px
        self.max_non_rigid_registration_dim_px = max_non_rigid_registration_dim_px
        self.max_micro_registration_dim_px = max_micro_registration_dim_px
        self.error_threshold_px = error_threshold_px
        self.max_attempts = max_attempts
        self.upscale_factor = upscale_factor
        self.confirm = confirm
        self.preview = preview
        self.adaptive_loop = adaptive_loop
        self.micro_registration = micro_registration
        self.crop = crop

        # CZI conversion control:
        # - [] or None: convert nothing
        # - [-1]: convert all CZIs
        # - [i, j, ...]: convert only those indices (0-based; 1-based accepted if no 0 present)
        self.transform_czi = list(transform_czi) if transform_czi is not None else []

        # For inspection / downstream use
        self.error_df = None
        self.error_metric_px = None
        self.applied_stage: Optional[str] = None
        self.results_root_dir: Optional[Path] = None

        super().__init__(metadata, channel_selection)

    @staticmethod
    def _validate_backend(rigid_transform, non_rigid_method, feature_detector, registration_stage):
        """Fail fast on unknown backend / stage choices (before VALIS is invoked)."""
        if rigid_transform not in RIGID_TRANSFORMS:
            raise ValueError(
                f"rigid_transform={rigid_transform!r} not in {sorted(RIGID_TRANSFORMS)}"
            )
        if non_rigid_method not in NON_RIGID_METHODS:
            raise ValueError(
                f"non_rigid_method={non_rigid_method!r} not in {sorted(NON_RIGID_METHODS)}"
            )
        if feature_detector not in FEATURE_DETECTORS:
            raise ValueError(
                f"feature_detector={feature_detector!r} not in {sorted(FEATURE_DETECTORS)}"
            )
        valid_stages = {"auto", "rigid", "non_rigid", "micro"}
        if registration_stage not in valid_stages:
            raise ValueError(
                f"registration_stage={registration_stage!r} not in {sorted(valid_stages)}"
            )

    def _valis_backend_kwargs(self) -> dict:
        """Translate the user's backend choices into registration.Valis kwargs."""
        kwargs = {
            "transformer_cls": RIGID_TRANSFORMS[self.rigid_transform],
        }

        nr_cls = NON_RIGID_METHODS[self.non_rigid_method]
        if nr_cls is None:
            # Rigid-only: disable non-rigid refinement entirely.
            kwargs["non_rigid_registrar_cls"] = None
        else:
            kwargs["non_rigid_registrar_cls"] = nr_cls()

        fd_cls = FEATURE_DETECTORS[self.feature_detector]
        if fd_cls is not None:
            # Classic detect-and-match: pair detector with a matching classic Matcher
            # so the LightGlue default (which only works with its own descriptors)
            # is not left mismatched against a hand-picked detector.
            kwargs["feature_detector_cls"] = fd_cls
            kwargs["matcher"] = feature_matcher.Matcher(feature_detector=fd_cls())

        return kwargs

    def _available_stages(self) -> list:
        """Stages that actually exist for the current backend configuration."""
        stages = ["rigid"]
        if self.non_rigid_method != "none":
            stages.append("non_rigid")
            if self.micro_registration:
                stages.append("micro")
        return stages

    def _resolve_stage(self, stage: str) -> str:
        """Map 'auto' to the best available stage; clamp an explicit choice."""
        available = self._available_stages()
        if stage == "auto" or stage not in available:
            return available[-1]
        return stage

    @staticmethod
    def _compute_registration_error(error_df) -> float:
        """
        Compute a single scalar error metric from VALIS error_df.

        Strategy:
        - Prefer non-rigid distance-based metrics ("non_rigid_*D").
        - Fallback to non-rigid TRE, then rigid metrics.
        - Use the median across all chosen values.
        - Return +inf if nothing usable is found.
        """
        import numpy as np

        if error_df is None or error_df.empty:
            return float("inf")

        cols = [c for c in error_df.columns if c.startswith("non_rigid") and c.endswith("D")]
        if not cols:
            cols = [c for c in error_df.columns if c.startswith("non_rigid") and c.endswith("TRE")]
        if not cols:
            cols = [c for c in error_df.columns if c.startswith("rigid") and c.endswith("D")]
        if not cols:
            cols = [c for c in error_df.columns if c.startswith("rigid") and c.endswith("TRE")]
        if not cols:
            return float("inf")

        vals = []
        for c in cols:
            v = np.asarray(error_df[c].values)
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(v)
        if not vals:
            return float("inf")

        vals = np.concatenate(vals)
        if vals.size == 0:
            return float("inf")
        return float(np.median(vals))

    def _run_single_registration(
            self,
            results_dir: Path,
            max_processed_dim: int,
            max_nonrigid_dim: int,
            max_micro_dim: int
    ):
        """
        Run one VALIS registration attempt at a given processed / non-rigid resolution.

        Returns
        -------
        registrar: registration.Valis
        error_df: pd.DataFrame
        temp_img_dir: Path
        registered_slide_dst_dir: Path
        """
        temp_img_dir = results_dir / "temp_imgs"
        temp_img_dir.mkdir(parents=True, exist_ok=True)

        # Prefer hardlinks / symlinks to avoid duplicating huge WSIs on disk
        for m in self.metadata:
            src = Path(m.path_storage)
            dst = temp_img_dir / src.name

            if dst.exists():
                console.print(f"[skip] {dst} already exists", style="info")
                continue

            try:
                # 1) try hard link first
                os.link(src, dst)
                console.print(f"[hardlink] {dst} -> {src}", style="info")

                # optional safety check: ensure no silent copy happened
                try:
                    if not src.samefile(dst):
                        console.print(
                            f"[warning] {dst} is not the same file as {src} after hardlink",
                            style="warning",
                        )
                except FileNotFoundError:
                    console.print(
                        f"[error] {dst} or {src} missing after hardlink attempt",
                        style="error",
                    )

            except OSError:
                try:
                    # 2) fallback: symlink if allowed
                    dst.symlink_to(src)
                    console.print(f"[symlink] {dst} -> {src}", style="info")

                    # optional safety check
                    try:
                        if not src.samefile(dst):
                            console.print(
                                f"[warning] {dst} is not the same file as {src} after symlink",
                                style="warning",
                            )
                    except FileNotFoundError:
                        console.print(
                            f"[error] {dst} or {src} missing after symlink attempt",
                            style="error",
                        )

                except OSError:
                    # 3) last resort: real copy
                    shutil.copy2(src, dst)
                    console.print(f"[copy] {src} -> {dst}", style="warning")

        registered_slide_dst_dir = results_dir / "registered_slides"
        registered_slide_dst_dir.mkdir(parents=True, exist_ok=True)

        registrar = registration.Valis(
            src_dir=str(temp_img_dir),
            dst_dir=str(results_dir),
            max_processed_image_dim_px=max_processed_dim,
            max_non_rigid_registration_dim_px=max_nonrigid_dim,
            imgs_ordered=self.imgs_ordered,
            crop=self.crop,
            **self._valis_backend_kwargs(),
        )

        try:
            rigid_registrar, non_rigid_registrar, error_df = registrar.register()

            # Micro registration refines the non-rigid field, so it only applies
            # when a non-rigid backend is active.
            if self.micro_registration and self.non_rigid_method != "none":
                registrar.register_micro(
                    max_non_rigid_registration_dim_px=max_micro_dim
                )

            error_df = registrar.measure_error()

            return registrar, error_df, temp_img_dir, registered_slide_dst_dir

        except Exception as e:
            console.print(
                f"[VALIS] registration attempt failed at "
                f"processed_dim={max_processed_dim}, non_rigid_dim={max_nonrigid_dim}: {e}",
                style="error",
            )
            # clean up temp images for this attempt
            shutil.rmtree(temp_img_dir, ignore_errors=True)
            shutil.rmtree(registered_slide_dst_dir, ignore_errors=True)
            # signal failure to caller
            return None, None, None, None

    def _ensure_ome_tiff_inputs(self, transform_czi: Optional[List[int]] = None) -> None:
        """
        Conditionally convert .czi inputs to OME-TIFF (written as sibling .tiff) and
        update Metadata.path_storage to point to the converted TIFF.

        Rules
        -----
        - If transform_czi is empty or None: convert nothing
        - If transform_czi contains -1: convert all CZIs
        - Else: treat entries as indices into self.metadata
          - default is 0-based
          - 1-based is accepted if there is no 0 in the list and all indices >= 1

        Keeps the original .czi on disk and keeps Metadata.path_original unchanged.
        """
        indices = list(transform_czi) if transform_czi is not None else []
        if not indices:
            console.print("[CZI->OME-TIFF] transform_czi is empty; skipping CZI conversion", style="info")
            return

        convert_all = (-1 in indices)

        # Determine target indices
        if convert_all:
            target_indices = list(range(len(self.metadata)))
            console.print("[CZI->OME-TIFF] -1 provided; converting all CZIs", style="info")
        else:
            raw = [int(i) for i in indices if i is not None]
            # Accept 1-based indexing if user did not include 0 and all are >= 1
            if raw and (0 not in raw) and min(raw) >= 1:
                raw = [i - 1 for i in raw]
            target_indices = sorted(set(raw))

        # Match Slidekick's standard CZI -> TIFF conversion parameters
        czi_kwargs = dict(tile_size=(2048, 2048), subresolution_levels=[1, 2, 4], res_unit="µm")

        for idx in target_indices:
            if idx < 0 or idx >= len(self.metadata):
                console.print(f"[CZI->OME-TIFF] index out of range: {idx} (n={len(self.metadata)})", style="warning")
                continue

            meta = self.metadata[idx]
            src = Path(meta.path_storage)

            if src.suffix.lower() != ".czi":
                console.print(f"[CZI->OME-TIFF] idx={idx} is not .czi ({src.name}); skipping", style="info")
                continue


            if not src.exists():
                console.print(f"[CZI->OME-TIFF] missing source file: {src}", style="error")
                continue

            # Store alongside the original CZI (same folder, same stem, .tiff)
            tiff_path = src.with_suffix(".tiff")

            if not tiff_path.exists():
                console.print(
                    f"[CZI->OME-TIFF] converting {src.name} -> {tiff_path.name}",
                    style="info",
                )
                try:
                    czi2tiff(src, tiff_path, **czi_kwargs)
                except Exception as e:
                    console.print(f"[CZI->OME-TIFF] conversion failed for {src.name}: {e}", style="error")
                    continue
            else:
                console.print(
                    f"[CZI->OME-TIFF] using existing {tiff_path.name} for {src.name}",
                    style="warning",
                )

            # Use the OME-TIFF for all downstream processing
            meta.path_storage = tiff_path
            meta.filename_stored = tiff_path.name
            meta.raw_format_stored = tiff_path.suffix

            # Optional: if metadata supports it, pull channel names/colors/calibration from OME-XML
            if hasattr(meta, "enrich_from_storage"):
                try:
                    meta.enrich_from_storage(overwrite=False)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Interactive napari preview
    # ------------------------------------------------------------------ #
    def _render_stage_images(self, registrar):
        """
        Build preview images per slide and per stage.

        Warped stages (rigid / non_rigid / micro) are rendered from real pyramid
        pixels at up to `preview_max_dim_px` on the long side (reading the
        smallest pyramid level >= that size, so warp cost stays bounded). The
        "original" reference and the pre-micro coarse non-rigid stay at thumbnail
        resolution. The full-resolution warp still only happens after Apply.

        Returns a list of dicts: {"name", "original"(RGB), "rigid"(gray),
        "non_rigid"(gray|None), "micro"(gray|None)}.
        """
        MAX_SIDE = int(self.preview_max_dim_px)
        warnings.filterwarnings(
            "ignore",
            message="scaling transformation for image with different shape",
            category=UserWarning,
        )

        def _downsample(img):
            if img is None:
                return None
            a = np.asarray(img)
            H, W = a.shape[:2]
            if max(H, W) <= MAX_SIDE:
                return a
            s = max(int(np.ceil(max(H, W) / float(MAX_SIDE))), 1)
            return a[::s, ::s] if a.ndim == 2 else a[::s, ::s, :]

        def _stretch99(x, invert=False):
            x = np.asarray(x).astype(np.float32, copy=False)
            if invert:  # brightfield: dark tissue on bright bg -> make tissue bright
                x = float(np.nanmax(x)) - x
            lo, hi = np.percentile(x, (1, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.nanmin(x)) if np.isfinite(np.nanmin(x)) else 0.0
                hi = float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else 1.0
                if hi <= lo:
                    hi = lo + 1.0
            y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
            return (y * 255.0).astype(np.uint8)

        def _to_gray(arr, is_bf):
            """Collapse to a single channel: luminance for brightfield, max-proj for fluorescence."""
            a = np.asarray(arr)
            if a.ndim == 2:
                return a
            if a.shape[2] == 1:
                return a[..., 0]
            if is_bf:
                return a[..., :3].astype(np.float32).mean(axis=2)  # luminance
            return a.max(axis=2)  # fluorescence: brightest channel wins

        def _original_gray(slide):
            """Unregistered reference as tissue-bright intensity (thumbnail res)."""
            is_bf = bool(getattr(slide, "is_rgb", False))
            src = getattr(slide, "image", None)
            if src is None:
                src = slide.processed_img
                is_bf = False  # processed_img is already fluorescence-like
            return _downsample(_stretch99(_to_gray(src, is_bf), invert=is_bf))

        def _coarse_non_rigid(slide):
            """Pre-micro coarse non-rigid from VALIS' saved thumbnail (always fluorescence-like)."""
            nr_path_str = getattr(slide, "nr_rigid_reg_img_f", None)
            if not nr_path_str:
                return None
            nr_path = Path(nr_path_str)
            if not nr_path.exists():
                return None
            try:
                v = pyvips.Image.new_from_file(str(nr_path), access="sequential")
                arr = warp_tools.vips2numpy(v)
                return _downsample(_stretch99(_to_gray(arr, is_bf=False)))
            except Exception:
                return None

        def _target_level(slide):
            """Smallest pyramid level whose long side is still >= MAX_SIDE."""
            dims = np.asarray(slide.slide_dimensions_wh)  # level 0 = full res
            long_sides = dims.max(axis=1)
            ok = np.where(long_sides >= MAX_SIDE)[0]
            return int(ok[-1]) if ok.size else 0

        def _warp_gray(slide, non_rigid):
            """Warp a real pyramid level, resize to MAX_SIDE, return tissue-bright intensity."""
            is_bf = bool(getattr(slide, "is_rgb", False))
            try:
                lvl = _target_level(slide)
                v = slide.warp_slide(level=lvl, non_rigid=non_rigid, crop=True)  # pyvips.Image
                long = max(v.width, v.height)
                if long > MAX_SIDE:
                    v = v.resize(MAX_SIDE / float(long))
                arr = warp_tools.vips2numpy(v)
                return _stretch99(_to_gray(arr, is_bf), invert=is_bf)
            except Exception:
                # Fallback to the low-res registration image if the level warp fails
                try:
                    return _downsample(_stretch99(
                        slide.warp_img(img=slide.processed_img, non_rigid=non_rigid, crop=True)))
                except Exception:
                    return None

        out = []
        n = len(registrar.slide_dict)
        for i, slide in enumerate(registrar.slide_dict.values(), start=1):
            console.print(f"[preview] rendering {slide.name} ({i}/{n}) at <= {MAX_SIDE}px", style="info")
            entry = {"name": slide.name, "original": _original_gray(slide),
                     "rigid": None, "non_rigid": None, "micro": None}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                entry["rigid"] = _warp_gray(slide, non_rigid=False)

                if self.non_rigid_method != "none":
                    if self.micro_registration:
                        # warp_slide(non_rigid=True) uses the micro-composed field.
                        # The pre-micro coarse non-rigid is only available from VALIS'
                        # saved thumbnail, so it stays at thumbnail resolution.
                        entry["non_rigid"] = _coarse_non_rigid(slide)
                        entry["micro"] = _warp_gray(slide, non_rigid=True)
                    else:
                        entry["non_rigid"] = _warp_gray(slide, non_rigid=True)
            out.append(entry)
        return out

    def _napari_preview(self, registrar):
        """
        Open an interactive napari viewer. Each slide/stage is a layer you can
        zoom, pan and toggle. A dock widget lets you switch the active stage and
        then Apply (warp the full-res slides with that stage) or Abort.

        Returns (do_apply: bool, stage: str).
        """
        import napari
        from magicgui.widgets import ComboBox, PushButton, Container, Label, SpinBox

        stages = self._available_stages()
        default_stage = self._resolve_stage(self.registration_stage)

        # Each slide gets its own color; additive blending makes co-registered
        # structures blend (toward white) and misalignment show as colored fringes.
        palette = ["green", "magenta", "cyan", "yellow", "red", "bop blue",
                   "bop orange", "blue"]

        viewer = napari.Viewer(title="VALIS registration preview")

        # Mutable layer bookkeeping, rebuilt whenever the preview px changes.
        state = {"stage_layers": {}, "view_layers": {}, "views": []}
        active = {"view": default_stage}
        result = {"apply": False, "stage": default_stage}

        def _build_layers(entries):
            """(Re)create one colored, additive layer per slide per view."""
            stage_layers = {s: [] for s in stages}
            original_layers = []
            for si, entry in enumerate(entries):
                name = entry["name"]
                cmap = palette[si % len(palette)]
                if entry["original"] is not None:
                    original_layers.append(
                        viewer.add_image(entry["original"], name=f"{name} · original",
                                         colormap=cmap, blending="additive", visible=True)
                    )
                for s in stages:
                    img = entry.get(s)
                    if img is None:
                        continue
                    stage_layers[s].append(
                        viewer.add_image(img, name=f"{name} · {s}", colormap=cmap,
                                         blending="additive", visible=False)
                    )
            view_layers = {}
            if original_layers:
                view_layers["original"] = original_layers
            for s in stages:
                view_layers[s] = stage_layers[s]
            state["stage_layers"] = stage_layers
            state["view_layers"] = view_layers
            state["views"] = list(view_layers.keys())

        def _show_view(view):
            if view not in state["view_layers"]:
                view = state["views"][-1]
            active["view"] = view
            for v, layers in state["view_layers"].items():
                for lyr in layers:
                    lyr.visible = (v == view)

        def _show_and_sync(view):
            _show_view(view)
            if active["view"] in stages and stage_combo.value != active["view"]:
                # keep the apply-stage selector in step with what you're viewing
                stage_combo.value = active["view"]

        def _cycle(_=None):
            views = state["views"]
            nxt = views[(views.index(active["view"]) + 1) % len(views)]
            _show_and_sync(nxt)

        # Build the first set of layers.
        entries = self._render_stage_images(registrar)
        if not entries:
            console.print("No slides for preview.", style="warning")
            viewer.close()
            return False, default_stage
        _build_layers(entries)
        _show_view(default_stage)
        viewer.reset_view()

        err = self.error_metric_px
        err_txt = f"{err:.2f} px" if (err is not None and np.isfinite(err)) else "n/a"

        info = Label(value=(f"median error: {err_txt}   |   backend: "
                            f"{self.rigid_transform}/{self.non_rigid_method}/{self.feature_detector}"))
        stage_combo = ComboBox(label="apply stage", choices=stages, value=default_stage)
        px_box = SpinBox(label="preview px", value=int(self.preview_max_dim_px),
                         min=32, max=16384, step=1)
        recompute_btn = PushButton(text="Recompute at this px")
        cycle_btn = PushButton(text=f"Cycle view  [{' › '.join(state['views'])}]   (C)")
        apply_btn = PushButton(text="Apply at full resolution")
        abort_btn = PushButton(text="Abort")

        def _recompute(_=None):
            new_px = int(px_box.value)
            self.preview_max_dim_px = new_px
            keep = active["view"]
            console.print(f"[preview] recomputing warped stages at <= {new_px}px ...", style="info")
            recompute_btn.enabled = False
            recompute_btn.text = "Recomputing ..."
            try:
                new_entries = self._render_stage_images(registrar)
                viewer.layers.clear()
                _build_layers(new_entries)
                _show_view(keep)
                viewer.reset_view()
            except Exception as e:
                console.print(f"[preview] recompute failed: {e}", style="error")
            finally:
                recompute_btn.enabled = True
                recompute_btn.text = "Recompute at this px"

        @stage_combo.changed.connect
        def _on_stage_change(value):
            result["stage"] = value
            _show_and_sync(value)

        recompute_btn.changed.connect(_recompute)
        cycle_btn.changed.connect(_cycle)

        @apply_btn.changed.connect
        def _on_apply(_=None):
            result["apply"] = True
            result["stage"] = stage_combo.value
            viewer.close()

        @abort_btn.changed.connect
        def _on_abort(_=None):
            result["apply"] = False
            viewer.close()

        # One-key cycling for speed; keeps zoom/pan intact between flips.
        @viewer.bind_key("c", overwrite=True)
        def _cycle_key(_viewer):
            _cycle()

        panel = Container(widgets=[info, stage_combo, px_box, recompute_btn,
                                   cycle_btn, apply_btn, abort_btn])
        viewer.window.add_dock_widget(panel, area="right", name="Registration")

        napari.run()  # blocks until the viewer is closed

        return result["apply"], result["stage"]

    def apply(self):
        """
        Run VALIS registration for the slides listed in self.metadata.

        Behavior
        --------
        - Uses Metadata objects as the primary interface.
        - Runs VALIS registration in one or more attempts, starting at
          max_processed_image_dim_px and max_non_rigid_registration_dim_px.
        - After each attempt, measures registration error from VALIS' error_df.
        - If the error is above error_threshold_px and attempts remain, it
          upsamples the working resolution and repeats.
        - Warps and saves full-resolution slides only for the best attempt.
        - Updates each Metadata.path_storage / path_original to point to the
          registered slides and sets image_type / uid accordingly.
                """
        # Optionally convert selected CZIs (controlled by transform_czi)
        self._ensure_ome_tiff_inputs(self.transform_czi)

        # Root results folder for all attempts
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        short_id = uuid.uuid4().hex[:8]
        new_uid = f"{timestamp}-{short_id}"

        results_root_dir = Path(OUTPUT_PATH) / f"valis_registration_{timestamp}_{short_id}"
        results_root_dir.mkdir(parents=True, exist_ok=True)
        self.results_root_dir = results_root_dir

        if self.adaptive_loop:

            # Adaptive resolution loop
            cur_proc_dim = int(self.max_processed_image_dim_px)
            cur_nonrigid_dim = int(self.max_non_rigid_registration_dim_px)
            cur_micro_dim = int(self.max_micro_registration_dim_px)

            best_metric = None
            best_registrar = None
            best_error_df = None
            best_results_dir = None
            best_temp_img_dir = None
            best_registered_slide_dst_dir = None

            for attempt in range(1, self.max_attempts + 1):
                attempt_results_dir = results_root_dir / f"attempt_{attempt}_proc{cur_proc_dim}_nr{cur_nonrigid_dim}"
                attempt_results_dir.mkdir(parents=True, exist_ok=True)

                console.print(
                    f"[VALIS] attempt {attempt}/{self.max_attempts} at "
                    f"processed_dim={cur_proc_dim}, non_rigid_dim={cur_nonrigid_dim}",
                    style="info",
                )

                registrar, error_df, temp_img_dir, registered_slide_dst_dir = self._run_single_registration(
                    attempt_results_dir,
                    cur_proc_dim,
                    cur_nonrigid_dim,
                    cur_micro_dim
                )

                if registrar is None:
                    # attempt failed, try next settings
                    cur_proc_dim = int(cur_proc_dim * self.upscale_factor)
                    cur_nonrigid_dim = int(cur_nonrigid_dim * self.upscale_factor)
                    cur_micro_dim = int(cur_micro_dim * self.upscale_factor)
                    continue

                metric = self._compute_registration_error(error_df)
                console.print(
                    f"[VALIS] attempt {attempt}: median registration error = {metric:.2f} px",
                    style="info",
                )

                # Track best attempt
                if best_metric is None or (np.isfinite(metric) and metric < best_metric):
                    best_metric = metric
                    best_registrar = registrar
                    best_error_df = error_df
                    best_results_dir = attempt_results_dir
                    best_temp_img_dir = temp_img_dir
                    best_registered_slide_dst_dir = registered_slide_dst_dir
                else:
                    # Not best, clean temp images and registered slides for this attempt
                    shutil.rmtree(temp_img_dir, ignore_errors=True)
                    shutil.rmtree(registered_slide_dst_dir, ignore_errors=True)
                    shutil.rmtree(attempt_results_dir, ignore_errors=True)
                    del registrar

                # Stop if good enough or last attempt
                if metric <= self.error_threshold_px or attempt == self.max_attempts:
                    break

                # Upsample for next attempt
                cur_proc_dim = int(cur_proc_dim * self.upscale_factor)
                cur_nonrigid_dim = int(cur_nonrigid_dim * self.upscale_factor)
                cur_micro_dim = int(cur_micro_dim * self.upscale_factor)

            # Use best attempt
            if best_registrar is None:
                console.print("VALIS registration failed for all attempts.", style="error")
                return self.metadata

            registrar = best_registrar
            error_df = best_error_df
            results_dir = best_results_dir
            temp_img_dir = best_temp_img_dir
            registered_slide_dst_dir = best_registered_slide_dst_dir

            self.error_df = error_df
            self.error_metric_px = best_metric

        else:
            # Single-shot registration (no adaptive loop) using the same helper
            proc_dim = int(self.max_processed_image_dim_px)
            nonrigid_dim = int(self.max_non_rigid_registration_dim_px)
            micro_dim = int(self.max_micro_registration_dim_px)

            # Mirror the attempt_* naming used in the adaptive loop
            results_dir = results_root_dir / f"attempt_1_proc{proc_dim}_nr{nonrigid_dim}"
            results_dir.mkdir(parents=True, exist_ok=True)

            console.print(
                f"[VALIS] single-shot registration at "
                f"processed_dim={proc_dim}, non_rigid_dim={nonrigid_dim}, micro_dim={micro_dim}",
                style="info",
            )

            # Delegate all the work (temp dir, copies, VALIS call, micro-reg, error_df)
            registrar, error_df, temp_img_dir, registered_slide_dst_dir = self._run_single_registration(
                results_dir,
                proc_dim,
                nonrigid_dim,
                micro_dim,  # if your helper only takes two dims, drop this argument
            )

            if registrar is None:
                console.print("VALIS registration failed.", style="error")
                return self.metadata

            # Keep the same bookkeeping as in the adaptive branch
            self.error_df = error_df
            self.error_metric_px = self._compute_registration_error(error_df)

        # ------------------------------------------------------------------ #
        # Interactive preview + choice of which stage to apply
        # ------------------------------------------------------------------ #
        default_stage = self._resolve_stage(self.registration_stage)
        chosen_stage = default_stage
        do_apply = True

        if self.preview:
            try:
                do_apply, chosen_stage = self._napari_preview(registrar)
            except Exception as e:
                console.print(
                    f"napari preview failed ({e}). Falling back to a text prompt.",
                    style="error",
                )
                if self.confirm:
                    do_apply = Confirm.ask(
                        f"Apply full-resolution transformation (stage={default_stage})?",
                        default=False, console=console,
                    )
        elif self.confirm:
            do_apply = Confirm.ask(
                f"Apply full-resolution transformation (stage={default_stage})?",
                default=False, console=console,
            )

        if not do_apply:
            console.print("Aborted by user. No warping performed.")
            return self.metadata

        chosen_stage = self._resolve_stage(chosen_stage)
        self.applied_stage = chosen_stage
        # "rigid" -> rigid-only warp; "non_rigid"/"micro" -> full non-rigid field
        # (the micro refinement, if computed, is already folded into that field).
        apply_non_rigid = chosen_stage != "rigid"
        console.print(f"[VALIS] applying stage '{chosen_stage}' at full resolution.", style="info")

        console.print(f"VALIS registration completed. Results directory: {results_dir}", style="info")

        input_files_used = registrar.original_img_list  # list[str]
        console.print(f"VALIS used these input files (in order it uses them): {input_files_used}", style="info")

        # Warp and save full-resolution slides to the registered_slide_dst_dir.
        # The VALIS API provides a method to warp & save the slides in native resolution.
        # We call that here so the full-resolution pyramid registered slides are available on disk.
        # NOTE: API method name in VALIS is `warp_and_save_slides` (per VALIS docs/examples).
        registrar.warp_and_save_slides(str(registered_slide_dst_dir), crop=self.crop, pyramid=True,
                                       colormap="auto", non_rigid=apply_non_rigid)

        console.print(f"Full-resolution registered slides saved to: {registered_slide_dst_dir}", style="info")

        # Delete temp dir
        shutil.rmtree(temp_img_dir)

        # VALIS saved the transformed images to the new path as individual objects

        # We update each metadata storage path to the transformed path
        # Normalized list of input paths as VALIS used them
        valis_inputs = [str(Path(p).resolve()) for p in registrar.original_img_list]  # now actually used

        # mapping input_path -> valis_name (VALIS-assigned short name)
        name_map = {str(Path(k).resolve()): v for k, v in registrar.name_dict.items()}

        # List all files VALIS wrote (in registered_slide_dst_dir)
        out_files = sorted(registered_slide_dst_dir.glob("*"))

        # Update each metadata entry
        for meta in self.metadata:
            # Normalize the path the same way we used when creating img_paths
            orig_path = str(Path(meta.path_storage).resolve())

            # Get valis_name from name_map (fallback: use stem)
            if orig_path in name_map:
                valis_name = name_map[orig_path]
            else:
                # fallback: try to match by stem among valis_inputs (handles relative/abs differences)
                stem_matches = [p for p in valis_inputs if Path(p).stem == Path(orig_path).stem]
                if stem_matches:
                    valis_name = name_map.get(stem_matches[0], Path(orig_path).stem)
                else:
                    valis_name = Path(orig_path).stem

            # find the registered file (robust to different extensions)
            new_file = find_registered_file_for_valis_name(registered_slide_dst_dir, valis_name, out_files)

            if new_file:
                # update metadata in-place (Path objects)
                meta.path_storage = new_file
                meta.path_original = new_file
                meta.image_type = "Registered WSI (VALIS)"
                meta.uid = f"{new_uid}-{valis_name}"
                console.print(f"Metadata updated: {meta.uid} -> {meta.path_storage.name}", style="info")
            else:
                console.print(f"No registered file found for VALIS name '{valis_name}' (meta uid {meta.uid})", style="error")

        del registrar  # Memory Management

        return self.metadata


# Helper: find file(s) that start with valis_name; prioritize common ome-tiff extensions
def find_registered_file_for_valis_name(registered_slide_dst_dir, valis_name, files):
    # try explicit expected extensions first
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        candidate = registered_slide_dst_dir / f"{valis_name}{ext}"
        if candidate.exists():
            return candidate
    # otherwise return first file whose name starts with valis_name
    prefix_matches = [f for f in files if f.name.startswith(valis_name)]
    return prefix_matches[0] if prefix_matches else None


if __name__ == "__main__":
    # Example usage
    from slidekick import DATA_PATH

    image_paths = [DATA_PATH / "reg" / "HE1.ome.tif",
                   DATA_PATH / "reg" / "HE2.ome.tif",
                   DATA_PATH / "reg" / "Arginase1.ome.tif",
                   DATA_PATH / "reg" / "KI67.ome.tif",
                   DATA_PATH / "reg" / "GS_CYP1A2.czi",
                   DATA_PATH / "reg" / "Ecad_CYP2E1.czi",
                   ]

    metadatas = [Metadata(path_original=image_path, path_storage=image_path) for image_path in image_paths]

    Registrator = ValisRegistrator(metadatas, max_processed_image_dim_px=600, max_non_rigid_registration_dim_px=600, max_micro_registration_dim_px=1200, micro_registration=True, transform_czi=[-1])

    metadatas_registered = Registrator.apply()
