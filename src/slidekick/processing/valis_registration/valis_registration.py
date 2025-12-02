from typing import List, Optional
import datetime
import uuid
from pathlib import Path
import shutil
from rich.prompt import Confirm
import warnings


from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH
from slidekick.processing.baseoperator import BaseOperator

# VALIS imports
from valis import registration

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pyvips

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
        error_threshold_px: float = 5.0,
        max_attempts: int = 3,
        upscale_factor: float = 1.5,
        confirm: bool = True,
        preview: bool = True,
        adaptive_loop: bool = False,
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
        error_threshold_px
            Target median feature-distance error in pixels. If the measured
            error is larger, the registration will be retried with upsampled
            processed / non-rigid dimensions.
        max_attempts
            Maximum number of registration attempts with increasing resolution.
        upscale_factor
            Multiplicative factor to increase processed / non-rigid dimensions
            between attempts.
        """
        self.save_img = save_img
        channel_selection = None
        self.imgs_ordered = imgs_ordered
        self.max_processed_image_dim_px = max_processed_image_dim_px
        self.max_non_rigid_registration_dim_px = max_non_rigid_registration_dim_px
        self.error_threshold_px = error_threshold_px
        self.max_attempts = max_attempts
        self.upscale_factor = upscale_factor
        self.confirm = confirm
        self.preview = preview
        self.adaptive_loop = adaptive_loop

        # For inspection / downstream use
        self.error_df = None
        self.error_metric_px = None
        self.results_root_dir: Optional[Path] = None

        super().__init__(metadata, channel_selection)

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
            max_nonrigid_dim: int
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

        for m in self.metadata:
            shutil.copy(m.path_storage, temp_img_dir)

        registered_slide_dst_dir = results_dir / "registered_slides"
        registered_slide_dst_dir.mkdir(parents=True, exist_ok=True)

        registrar = registration.Valis(
            src_dir=str(temp_img_dir),
            dst_dir=str(results_dir),
            max_processed_image_dim_px=max_processed_dim,
            max_non_rigid_registration_dim_px=max_nonrigid_dim,
            imgs_ordered=self.imgs_ordered,
            crop="reference",
        )

        try:
            rigid_registrar, non_rigid_registrar, error_df = registrar.register()

            registrar.register_micro(
                max_non_rigid_registration_dim_px=max_nonrigid_dim
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
                    cur_nonrigid_dim
                )

                if registrar is None:
                    # attempt failed, try next settings
                    cur_proc_dim = int(cur_proc_dim * self.upscale_factor)
                    cur_nonrigid_dim = int(cur_nonrigid_dim * self.upscale_factor)
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
                    # Not best, clean temp images for this attempt
                    shutil.rmtree(temp_img_dir, ignore_errors=True)
                    del registrar

                # Stop if good enough or last attempt
                if metric <= self.error_threshold_px or attempt == self.max_attempts:
                    break

                # Upsample for next attempt
                cur_proc_dim = int(cur_proc_dim * self.upscale_factor)
                cur_nonrigid_dim = int(cur_nonrigid_dim * self.upscale_factor)

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
            temp_img_dir = results_root_dir / "temp_imgs"
            temp_img_dir.mkdir(parents=True, exist_ok=True)

            # Copy every image metadata from path_storage to temp_img_dir
            for m in self.metadata:
                shutil.copy(m.path_storage, temp_img_dir)

            # registered slides destination folder
            registered_slide_dst_dir = results_root_dir / "registered_slides"
            registered_slide_dst_dir.mkdir(parents=True, exist_ok=True)

            # VALIS parameter:
            # - max_processed_image_dim_px controls the maximum dimension of the images used
            #   for the 'processed' image (used to find transforms). By setting it very large,
            #   VALIS will effectively use the highest-resolution level for transform estimation.

            # Initialize VALIS registrar with the list of slide image paths.
            # imgs_ordered=False lets VALIS reorder the images by similarity (unordered structure).
            # reference_img_f=None leaves selection of reference image to VALIS (it will pick center).

            registrar = registration.Valis(
                src_dir=str(temp_img_dir),
                dst_dir=str(results_root_dir),
                # img_list=img_paths, Unused for now
                max_processed_image_dim_px=self.max_processed_image_dim_px,
                max_non_rigid_registration_dim_px=self.max_non_rigid_registration_dim_px,
                imgs_ordered=self.imgs_ordered,
                crop="reference"
            )

            # Run registration: returns rigid and non-rigid registrar objects and an error dataframe
            rigid_registrar, non_rigid_registrar, error_df = registrar.register()
            registrar.register_micro(max_non_rigid_registration_dim_px=self.max_non_rigid_registration_dim_px)

            results_dir = self.results_root_dir

        # Preview: rows = [original color, rigid(gray), non-rigid(gray)] x cols = slides
        if self.preview:
            try:
                slides = list(registrar.slide_dict.values())
                if not slides:
                    console.print("No slides for preview.", style="error")
                    return self.metadata

                MAX_SIDE = 2048
                warnings.filterwarnings(
                    "ignore",
                    message="scaling transformation for image with different shape",
                    category=UserWarning,
                )

                def _downsample_stride(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
                    if img is None:
                        return None
                    a = np.asarray(img)
                    H, W = a.shape[:2]
                    if max(H, W) <= max_side:
                        return a
                    s = max(int(np.ceil(max(H, W) / float(max_side))), 1)
                    return a[::s, ::s] if a.ndim == 2 else a[::s, ::s, :]

                def _stretch99(x: np.ndarray) -> np.ndarray:
                    x = x.astype(np.float32, copy=False)
                    lo, hi = np.percentile(x, (1, 99))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo = np.nanmin(x) if np.isfinite(np.nanmin(x)) else 0.0
                        hi = np.nanmax(x) if np.isfinite(np.nanmax(x)) else 1.0
                        if hi <= lo:
                            hi = lo + 1.0
                    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
                    return (y * 255.0).astype(np.uint8)

                def _guess_brightfield(arr: np.ndarray) -> bool:
                    a = np.asarray(arr)
                    return a.ndim == 3 and a.shape[2] >= 3 and a.dtype == np.uint8

                def _rgb_native(a: np.ndarray) -> np.ndarray:
                    """Preserve brightfield colors. Take first 3 channels. No per-channel stretch."""
                    x = np.asarray(a)
                    if x.ndim == 2:  # rare for BF; make neutral RGB
                        x = np.repeat(x[:, :, None], 3, axis=2)
                    else:
                        x = x[..., :3]
                    if x.dtype == np.uint8:
                        return x
                    if x.dtype == np.uint16:
                        return (x / 257.0).clip(0, 255).astype(np.uint8)
                    # float or other: uniform scale by global max (keeps color balance)
                    x = x.astype(np.float32, copy=False)
                    m = np.nanmax(x)
                    m = 1.0 if (not np.isfinite(m) or m <= 0) else m
                    return np.clip(x / m * 255.0, 0, 255).astype(np.uint8)

                def _pick_rgb3(src: np.ndarray) -> list[np.ndarray]:
                    """Choose 3 channels for fluorescence. Handles 2D, 3D, ND."""
                    x = np.asarray(src)
                    if x.ndim == 2:
                        return [x, x, x]
                    if x.ndim == 3:
                        C = x.shape[2]
                        if C >= 3:
                            return [x[..., 0], x[..., 1], x[..., 2]]
                        if C == 2:
                            c0, c1 = x[..., 0], x[..., 1]
                            return [c0, c1, np.maximum(c0, c1)]
                        g = x[..., 0]
                        return [g, g, g]
                    # ND fallback: flatten non-spatial dims to channels, take first three
                    order = np.argsort(x.shape)[::-1]
                    y_dim, x_dim = order[:2]
                    y = np.transpose(x, (y_dim, x_dim, *[d for d in range(x.ndim) if d not in (y_dim, x_dim)]))
                    H, W = y.shape[:2]
                    C = int(np.prod(y.shape[2:])) if y.ndim > 2 else 1
                    y = y.reshape(H, W, C)
                    if C >= 3:
                        return [y[..., 0], y[..., 1], y[..., 2]]
                    if C == 2:
                        c0, c1 = y[..., 0], y[..., 1]
                        return [c0, c1, np.maximum(c0, c1)]
                    g = y[..., 0]
                    return [g, g, g]

                def _rgb_from_channels_stretched(chs: list[np.ndarray]) -> np.ndarray:
                    """Fluorescence: per-channel percentile stretch to avoid black tiles."""
                    return np.stack([_stretch99(np.asarray(chs[0])),
                                     _stretch99(np.asarray(chs[1])),
                                     _stretch99(np.asarray(chs[2]))], axis=-1)

                # Map VALIS short name -> original file path (for thumbnail fallback)
                name_to_src = {v: k for k, v in registrar.name_dict.items()}

                def _read_original_color(slide) -> np.ndarray:
                    # Prefer VALIS-provided slide.image
                    src = getattr(slide, "image", None)
                    if src is not None:
                        if _guess_brightfield(src):
                            return _downsample_stride(_rgb_native(src))
                        # fluorescence or non-uint8 3-channel -> stretch per channel
                        return _downsample_stride(_rgb_from_channels_stretched(_pick_rgb3(src)))
                    # Fallback: thumbnail from original file via pyvips
                    src_fp = name_to_src.get(slide.name, None)
                    if src_fp and Path(src_fp).exists():
                        try:
                            v = pyvips.Image.thumbnail(str(src_fp), MAX_SIDE)
                            arr = np.frombuffer(v.write_to_memory(), dtype=np.uint8).reshape(v.height, v.width, v.bands)
                            if arr.shape[2] == 1:
                                arr = np.repeat(arr, 3, axis=2)
                            if arr.shape[2] > 3:
                                arr = arr[..., :3]
                            return arr
                        except Exception:
                            pass
                    # Last resort: processed gray replicated
                    g = _stretch99(slide.processed_img)
                    return np.stack([g, g, g], axis=-1)

                per_slide = []
                for slide in slides:
                    name = slide.name
                    proc = slide.processed_img  # registration resolution (2D)

                    # Row 1: original color (BF native, FL stretch), ≤2048
                    orig_rgb = _read_original_color(slide)

                    # Rows 2–3: warp processed image to grayscale previews (shape-matched)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        try:
                            rigid_gray = slide.warp_img(img=proc, non_rigid=False, crop=True)
                        except Exception:
                            rigid_gray = None
                        try:
                            nonrigid_gray = slide.warp_img(img=proc, non_rigid=True, crop=True)
                        except Exception:
                            nonrigid_gray = None

                    rigid_gray = _downsample_stride(_stretch99(rigid_gray)) if rigid_gray is not None else None
                    nonrigid_gray = _downsample_stride(_stretch99(nonrigid_gray)) if nonrigid_gray is not None else None

                    per_slide.append((name, orig_rgb, rigid_gray, nonrigid_gray))

                if not per_slide:
                    console.print("Nothing to preview.", style="warning")
                    return self.metadata

                # Plot 3 rows x N columns
                n = len(per_slide)
                fig_w = max(3.0 * n, 6.0)
                fig_h = 9.0
                fig, axes = plt.subplots(3, n, figsize=(fig_w, fig_h), constrained_layout=True)
                if n == 1:
                    axes = np.expand_dims(axes, 1)

                row_titles = ["original color", "rigid", "non-rigid"]
                for j, (name, orig_p, rigid_p, nonrigid_p) in enumerate(per_slide):
                    axes[0, j].set_title(name, fontsize=10)

                    # row 1: color
                    ax0 = axes[0, j]
                    ax0.imshow(orig_p) if orig_p is not None else ax0.text(0.5, 0.5, "N/A", ha="center", va="center",
                                                                           fontsize=9)
                    ax0.set_ylabel(row_titles[0], rotation=90, ha="right", va="center", fontsize=10, labelpad=18)
                    ax0.set_xticks([]);
                    ax0.set_yticks([]);
                    [s.set_visible(False) for s in ax0.spines.values()]

                    # row 2: rigid grayscale
                    ax1 = axes[1, j]
                    ax1.imshow(rigid_p, cmap="gray") if rigid_p is not None else ax1.text(0.5, 0.5, "N/A", ha="center",
                                                                                          va="center", fontsize=9)
                    ax1.set_ylabel(row_titles[1], rotation=90, ha="right", va="center", fontsize=10, labelpad=18)
                    ax1.set_xticks([]);
                    ax1.set_yticks([]);
                    [s.set_visible(False) for s in ax1.spines.values()]

                    # row 3: non-rigid grayscale
                    ax2 = axes[2, j]
                    ax2.imshow(nonrigid_p, cmap="gray") if nonrigid_p is not None else ax2.text(0.5, 0.5, "N/A",
                                                                                                ha="center",
                                                                                                va="center", fontsize=9)
                    ax2.set_ylabel(row_titles[2], rotation=90, ha="right", va="center", fontsize=10, labelpad=18)
                    ax2.set_xticks([]);
                    ax2.set_yticks([]);
                    [s.set_visible(False) for s in ax2.spines.values()]

                plt.show()

            except Exception as e:
                console.print(f"Preview failed ({e}). Continuing without preview.", style="error")

        if self.confirm:
            apply = Confirm.ask("Apply full-resolution transformation and save outputs?", default=False,
                                console=console)
            if not apply:
                console.print("Aborted by user. No warping performed.")
                return self.metadata

        console.print(f"VALIS registration completed. Results directory: {results_dir}", style="info")

        input_files_used = registrar.original_img_list  # list[str]
        console.print(f"VALIS used these input files (in order it uses them): {input_files_used}", style="info")

        # Warp and save full-resolution slides to the registered_slide_dst_dir.
        # The VALIS API provides a method to warp & save the slides in native resolution.
        # We call that here so the full-resolution registered slides are available on disk.
        #
        # NOTE: API method name in VALIS is `warp_and_save_slides` (per VALIS docs/examples).
        #       If you want a different cropping method, set crop="reference" or crop="overlap".
        registrar.warp_and_save_slides(str(registered_slide_dst_dir), crop="overlap")

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

    Registrator = ValisRegistrator(metadatas, max_processed_image_dim_px=600, max_non_rigid_registration_dim_px=600)

    metadatas_registered = Registrator.apply()
