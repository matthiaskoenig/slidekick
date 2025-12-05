import numpy as np
from pathlib import Path
from rich.prompt import Confirm, Prompt
import zarr

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.io import save_tif
from slidekick.console import console
from slidekick import OUTPUT_PATH

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# presets and aliases
PROFILE_ALIASES = {
    "HE": "H&E", "H_E": "H&E", "H-E": "H&E", "H&E": "H&E",
    "HDAB": "H-DAB", "H-DAB": "H-DAB", "H_DAB": "H-DAB",
    "HED": "HED",
    "Arginase1": "Arginase1", "Arginase": "Arginase1",
}

# Columns are stain OD vectors (R,G,B)^T
DEFAULT_VECTORS = {
    "H&E": np.array([[0.650, 0.072],
                     [0.704, 0.990],
                     [0.286, 0.105]], dtype=float),
    "H-DAB": np.array([[0.650, 0.268],
                       [0.704, 0.570],
                       [0.286, 0.776]], dtype=float),
    "HED": np.array([[0.650, 0.072, 0.268],
                     [0.704, 0.990, 0.570],
                     [0.286, 0.105, 0.776]], dtype=float),
    "Arginase1": np.array([[-0.762590, 0.548268, 0.343306],
                           [0.644835, 0.602108, 0.470801],
                           [0.051418, 0.580404, -0.812704]], dtype=float),
}
DEFAULT_STAIN_NAMES = {
    "H&E": ["Hematoxylin", "Eosin"],
    "H-DAB": ["Hematoxylin", "DAB"],
    "HED": ["Hematoxylin", "Eosin", "DAB"],
    "Arginase1": ["Arginase1", "NuclearFastRed"],
}

# helpers
def _estimate_channel_bg(ch_arr: np.ndarray,
                         n_bins: int = 512) -> float:
    """
    Estimate the background level of a single image channel.

    Parameters
    ----------
    ch_arr : np.ndarray
        2D image array containing the channel intensities.
    n_bins : int, optional
        Number of bins to use for the histogram, by default 512.

    Returns
    -------
    float
        Estimated background intensity level.
    """
    # Flatten the array and ensure float32 for consistency
    vals = ch_arr.ravel().astype(np.float32)

    # Remove NaN and inf values, so they do not affect histogram statistics
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0

    # Lower limit is the minimum observed value
    lo = float(vals.min())

    # Upper limit is the 99th percentile to avoid extreme bright outliers
    hi = float(np.percentile(vals, 99.0))

    # Ensure the histogram range is not degenerate
    if hi <= lo:
        hi = lo + 1.0

    # Compute histogram over the chosen range
    hist, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Peak detection
    # Interior local maxima:
    #   hist[i] is a peak if it is strictly greater than the previous bin
    #   and greater or equal to the next bin.
    interior_peaks = np.where(
        (hist[1:-1] > hist[:-2]) &
        (hist[1:-1] >= hist[2:])
    )[0] + 1  # shift back to original indices

    # Handle edge bins explicitly, since the expression above never tests
    # hist[0] or hist[-1]. This is why the original code missed the peak
    # at zero.
    peak_indices = list(interior_peaks)

    # Left edge: consider it a peak if it is not lower than its only neighbor
    if hist.size > 1 and hist[0] >= hist[1]:
        peak_indices.insert(0, 0)

    # Right edge: consider it a peak if it is not lower than its only neighbor
    if hist.size > 1 and hist[-1] >= hist[-2]:
        peak_indices.append(len(hist) - 1)

    peak_idx = np.array(peak_indices, dtype=int)

    # If no peaks are found (highly unlikely but possible for flat histograms),
    # fall back to a robust statistic.
    if peak_idx.size == 0:
        bg_level = float(np.percentile(vals, 5.0))
    else:
        # Sort peaks from low to high intensity
        peak_idx = peak_idx[np.argsort(centers[peak_idx])]

        # Filter out very small peaks: keep only peaks that have at least
        # a certain fraction of the global maximum height
        height_thresh = 0.02 * hist.max()
        peak_idx = [i for i in peak_idx if hist[i] >= height_thresh]

        # If there are still no peaks after thresholding, fall back to median
        if len(peak_idx) == 0:
            bg_level = float(np.percentile(vals, 50.0))
        else:
            # If we have at least three peaks, we assume:
            #   peak_idx[0]  big spike at (or near) zero
            #   peak_idx[1]  background mode
            # Otherwise we just take the first peak.
            if len(peak_idx) >= 3:
                idx_bg = peak_idx[1]
            else:
                idx_bg = peak_idx[0]

            bg_level = float(centers[idx_bg])

    return float(bg_level)

def _to_numpy(arr):
    """Coerce zarr NumPy-like arrays to NumPy ndarray."""
    if isinstance(arr, zarr.Array):
        return arr[...]
    else:
        return np.asarray(arr)


def _normalize_image_01(img_np: np.ndarray) -> np.ndarray:
    """Return float32 in [0,1] for brightfield math. Handles uint8/uint16/float."""
    src_dtype = img_np.dtype
    arr = img_np.astype(np.float32, copy=False)
    if np.issubdtype(src_dtype, np.integer):
        maxv = float(np.iinfo(src_dtype).max)
        if maxv <= 0:
            maxv = 255.0
        arr = arr / maxv
    arr = np.clip(arr, 1e-8, 1.0)  # avoid log(0)
    return arr


def _first_available_level(level_dicts):
    """Return a level key present across stains, prefer smallest image (highest level index)."""
    keys_sets = [set(d.keys()) for d in level_dicts if d]
    if not keys_sets:
        return None
    common = set.intersection(*keys_sets) if len(keys_sets) > 1 else keys_sets[0]
    if not common:
        for d in level_dicts:
            if d:
                return max(d.keys())
        return None
    return max(common)


def _iter_tiles(arr, tile_size: int = 2048):
    """
    Iterate over 2D tiles of an array with shape (H, W) or (H, W, C).
    Yields pairs of slice objects (ys, xs).
    """
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError(f"Unsupported shape for tiling: {shape}")
    H, W = shape[0], shape[1]
    for y0 in range(0, H, tile_size):
        y1 = min(y0 + tile_size, H)
        ys = slice(y0, y1)
        for x0 in range(0, W, tile_size):
            x1 = min(x0 + tile_size, W)
            xs = slice(x0, x1)
            yield ys, xs


class StainSeparator(BaseOperator):

    def __init__(self,
                 metadata,
                 channel_selection=None,
                 mode: str = None,
                 stain_profile: str = None,
                 custom_matrix: np.ndarray = None,
                 confirm: bool = True,
                 preview: bool = True,
                 remove_background: bool = True):
        """
        - mode: "brightfield" or "fluorescence" (auto if None)
        - stain_profile: e.g. "H&E", "H-DAB" (brightfield only)
        - custom_matrix: optional 3xN stain OD matrix (brightfield)
        - remove_background: remove background in flourescence
        """
        self.mode = mode
        self.stain_profile = stain_profile
        self.custom_matrix = custom_matrix
        self.confirm = confirm
        self.preview = preview
        self.remove_background = remove_background
        super().__init__(metadata, channel_selection=None)

    def apply(self):
        """Execute stain separation on the input metadata."""
        mode = self.mode
        if mode is None:
            try:
                img_example = self.load_image()
                sample_level = next(iter(img_example.keys()))
                arr0 = img_example[sample_level]
                shape = getattr(arr0, "shape", None)
                ndim = len(shape) if shape is not None else 0
                nchan = shape[2] if ndim == 3 else 1
                itype = str(getattr(self.metadata[0], "image_type", "")).lower()
                if ndim == 3 and nchan == 3 and itype in {"brightfield", "immunohistochemistry"}:
                    mode = "brightfield"
                elif ndim == 3 and nchan > 3:
                    mode = "fluorescence"
                else:
                    mode = "fluorescence" if ("fluorescence" in itype or "fluor" in itype) else "brightfield"
            except Exception as e:
                console.print(f"Could not determine image modality automatically: {e}", style="warning")
                mode = "brightfield"
        console.print(f"StainSeparator mode: {mode}", style="info")
        if mode == "brightfield":
            return self._apply_brightfield()
        else:
            return self._apply_fluorescence()

    # brightfield
    # brightfield
    def _apply_brightfield(self):
        """
        Brightfield color deconvolution with:
        - tiled processing (no huge H*W flatten)
        - full pyramid preserved (all input levels represented)
        - level->ndarray dict passed to save_tif, so downstream readers see a proper image stack.
        """
        # 1) Resolve profile and normalize stain matrix
        M, stain_names, profile = self._resolve_brightfield_profile()

        # normalize columns and add residual if needed (float32 to reduce memory)
        M = np.array(
            [c / (np.linalg.norm(c) + 1e-8) for c in M.T],
            dtype=np.float32,
        ).T
        if M.shape[1] == 2:
            v1, v2 = M[:, 0], M[:, 1]
            v3 = np.cross(v1, v2)
            n = np.linalg.norm(v3)
            if n < 1e-6:
                raise ValueError(
                    f"Provided stain vectors are nearly collinear for profile {profile}."
                )
            v3 /= n
            M = np.column_stack([M, v3.astype(np.float32)])
            stain_names = stain_names + ["Residual"]

        # precompute deconvolution matrix once
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        M_inv = M_inv.astype(np.float32)

        # 2) Read all pyramid levels
        image = self.load_image()  # dict[level] -> array
        levels = sorted(image.keys())

        # 3) Preview on a downsampled level (optional, safe)
        if self.preview:
            try:
                preview_level = max(levels)
                img_prev = _to_numpy(image[preview_level])
                if img_prev.ndim == 3 and img_prev.shape[2] >= 3:
                    Hprev, Wprev = img_prev.shape[:2]
                    stride = max(int(np.ceil(max(Hprev, Wprev) / 2048)), 1)
                    img_prev_small = img_prev[::stride, ::stride, :]

                    arr01_prev = _normalize_image_01(img_prev_small)
                    OD_prev = -np.log10(arr01_prev.astype(np.float32, copy=False))
                    C_prev = OD_prev @ M_inv.T  # (h, w, n_stains)

                    color_panels = []
                    names_for_preview = []

                    n_preview = min(C_prev.shape[2], len(stain_names))
                    for j in range(n_preview):
                        name_j = stain_names[j]
                        if name_j.lower() == "residual":
                            continue
                        ODj = C_prev[:, :, j]
                        col = M[:, j]  # (3,)
                        OD_rgb = ODj[..., None] * col[None, None, :]
                        I_rgb = np.power(10.0, -np.clip(OD_rgb, 0.0, None))
                        rgb = (np.clip(I_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                        color_panels.append(rgb)
                        names_for_preview.append(name_j)

                    total = 1 + len(color_panels)
                    cols = min(total, 3)
                    rows = int(np.ceil(total / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
                    axes = np.atleast_1d(axes).ravel()

                    axes[0].imshow(
                        img_prev_small
                        if img_prev_small.shape[-1] == 3
                        else img_prev_small[..., :3]
                    )
                    axes[0].set_title("Original")
                    axes[0].axis("off")

                    for ax, panel, name in zip(axes[1:], color_panels, names_for_preview):
                        ax.imshow(panel)
                        ax.set_title(name)
                        ax.axis("off")

                    for ax in axes[1 + len(color_panels):]:
                        ax.axis("off")

                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                console.print(f"Preview generation failed: {e}", style="error")

        # 4) Confirm before heavy work
        if self.confirm:
            apply_full = Confirm.ask(
                "Apply stain separation to full resolution image(s)?",
                default=True,
                console=console,
            )
            if not apply_full:
                console.print("Stain separation aborted by user.", style="warning")
                return self.metadata

        # 5) Prepare output paths and temporary disk-backed storage
        base_path = Path(self.metadata[0].path_storage)
        base_stem = (
            base_path.name[:-len("".join(base_path.suffixes))]
            if base_path.suffixes
            else base_path.stem
        )

        dest_dir = Path(OUTPUT_PATH) / self.metadata[0].uid
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = dest_dir / "_tmp_mm_brightfield"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        n_stains = M.shape[1]
        separated_levels = [dict() for _ in range(n_stains)]
        dtype_out = np.uint8
        tile_size = 2048

        # 6) Full-resolution, tiled processing for each level
        for level in levels:
            img_np = _to_numpy(image[level])

            # if level is not 3-channel, we mark placeholders and fill later
            if img_np.ndim != 3 or img_np.shape[2] < 3:
                console.print(
                    f"Level {level} image is not 3-channel, "
                    f"will reuse nearest separated level as placeholder.",
                    style="warning",
                )
                for j in range(n_stains):
                    # mark as None, will be filled after processing all levels
                    if level not in separated_levels[j]:
                        separated_levels[j][level] = None
                continue

            H, W, _ = img_np.shape

            # allocate one memmap per stain for this level
            mmaps = []
            for j in range(n_stains):
                mm_path = tmp_dir / f"{base_stem}_L{level}_stain{j}.dat"
                mm = np.memmap(mm_path, mode="w+", dtype=dtype_out, shape=(H, W))
                separated_levels[j][level] = mm
                mmaps.append(mm)

            # tiled processing
            for ys, xs in _iter_tiles(img_np, tile_size=tile_size):
                tile = img_np[ys, xs, :].astype(np.float32, copy=False)
                arr01 = _normalize_image_01(tile)
                OD_tile = -np.log10(arr01)
                C_tile = OD_tile @ M_inv.T  # (tile_h, tile_w, n_stains)

                for j, mm in enumerate(mmaps):
                    ODj = C_tile[:, :, j]
                    Ij = np.power(10.0, -ODj)
                    out_tile = (np.clip(Ij, 0.0, 1.0) * 255.0).astype(dtype_out)
                    mm[ys, xs] = out_tile

        # 7) Fill placeholder levels (non-3-channel) by copying nearest real level
        for j in range(n_stains):
            stain_levels = separated_levels[j]
            if not stain_levels:
                continue

            real_levels = [lvl for lvl, arr in stain_levels.items() if arr is not None]
            if not real_levels:
                continue

            for lvl in levels:
                if stain_levels.get(lvl) is None:
                    nearest = min(real_levels, key=lambda L: abs(L - lvl))
                    ref = stain_levels[nearest]
                    stain_levels[lvl] = np.array(ref, copy=True)

        # 8) Save outputs and create metadata per stain
        output_metadata = []
        for j, stain_name in enumerate(stain_names):
            if stain_name.lower() == "residual":
                continue

            out_name = f"{base_stem}_{stain_name.replace(' ', '_')}.tiff"
            out_path = dest_dir / out_name

            new_meta = Metadata(
                path_original=self.metadata[0].path_original,
                path_storage=out_path,
                image_type=self.metadata[0].image_type,
                uid=f"{self.metadata[0].uid}-{stain_name.replace(' ', '_')}",
            )
            new_meta.set_stains({0: stain_name})
            new_meta.save(dest_dir)

            # ensure we pass a clean level->ndarray dict, with sorted levels
            level_dict = separated_levels[j]
            clean_levels = {
                lvl: np.asarray(level_dict[lvl])
                for lvl in sorted(level_dict.keys())
            }

            # Passing a level→array dict => save_tif writes tiled, pyramidal TIFF
            save_tif(clean_levels, out_path, metadata=new_meta)

            console.print(f"Saved [{stain_name}] to {out_path}", style="success")
            output_metadata.append(new_meta)

        del image, separated_levels  # Memory Management
        return output_metadata

    # profile resolver with no silent defaults
    def _resolve_brightfield_profile(self):
        """Return (M, stain_names, profile_name). Never silently default."""
        meta = self.metadata[0] if isinstance(self.metadata, (list, tuple)) else self.metadata

        # 1) explicit custom matrix
        if getattr(self, "custom_matrix", None) is not None:
            M = np.asarray(self.custom_matrix, dtype=float)
            names = [f"Stain{i}" for i in range(M.shape[1])]
            return M, names, "CUSTOM"

        # 2) explicit profile param
        prof = getattr(self, "stain_profile", None)
        if prof:
            prof = PROFILE_ALIASES.get(prof, prof)
            if prof not in DEFAULT_VECTORS:
                raise ValueError(
                    f"Unknown stain_profile '{prof}'. Options: {list(DEFAULT_VECTORS)} or provide custom_matrix.")
            return DEFAULT_VECTORS[prof], DEFAULT_STAIN_NAMES[prof], prof

        # 3) metadata hints (non-fatal)
        stains = getattr(meta, "stains", None) or {}
        stain_names_l = {str(v).strip().lower() for v in stains.values()}
        if {"hematoxylin", "eosin"} <= stain_names_l:
            return DEFAULT_VECTORS["H&E"], DEFAULT_STAIN_NAMES["H&E"], "H&E"
        if "dab" in stain_names_l and "hematoxylin" in stain_names_l:
            return DEFAULT_VECTORS["H-DAB"], DEFAULT_STAIN_NAMES["H-DAB"], "H-DAB"

        itype = (getattr(meta, "image_type", "") or "").lower()
        fname = Path(getattr(meta, "path_storage", "") or "").name.lower()
        if any(k in itype + " " + fname for k in ["ihc", "dab", "h-dab", "hdab"]):
            return DEFAULT_VECTORS["H-DAB"], DEFAULT_STAIN_NAMES["H-DAB"], "H-DAB"
        if any(k in itype + " " + fname for k in ["he", "h&e"]):
            return DEFAULT_VECTORS["H&E"], DEFAULT_STAIN_NAMES["H&E"], "H&E"

        # 4) interactive choice or hard fail
        options = list(DEFAULT_VECTORS.keys()) + ["CUSTOM"]
        if getattr(self, "confirm", True):
            choice = Prompt.ask("Select brightfield stain profile", choices=options, default="H&E")
            if choice == "CUSTOM":
                raise ValueError("CUSTOM chosen but no custom_matrix provided.")
            return DEFAULT_VECTORS[choice], DEFAULT_STAIN_NAMES[choice], choice

        raise ValueError("Brightfield stain profile required. Set stain_profile or custom_matrix.")

    def _apply_fluorescence(self):
        """Preview is downsampled only. Saved outputs are full resolution.
        Preview shows an 'Original' panel (true RGB if C==3, else gray MIP) plus colorized channels.
        If C==3, channel colors are R/G/B to match the image's own colors; else use a palette.
        """
        # helpers
        MAX_PREVIEW_SIDE = 2048  # preview-only cap
        MAX_BG_SIDE = 2048  # cap for background estimation

        PALETTE = np.array([
            [1.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0],  # green
            [0.0, 0.0, 1.0],  # blue
            [1.0, 0.0, 1.0],  # magenta
            [0.0, 1.0, 1.0],  # cyan
            [1.0, 1.0, 0.0],  # yellow
            [1.0, 0.5, 0.0],  # orange
            [0.6, 0.0, 1.0],  # violet
            [0.0, 0.7, 0.3],  # teal
            [1.0, 0.6, 0.6],  # pink
        ], dtype=np.float32)

        def _stretch99(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float32, copy=False)
            p99 = float(np.percentile(x, 99.0)) if x.size else 1.0
            d = p99 if p99 > 0 else (float(x.max()) if x.max() > 0 else 1.0)
            return np.clip(x / d, 0.0, 1.0)

        def _colorize(ch2d: np.ndarray, rgb: np.ndarray) -> np.ndarray:
            y = _stretch99(ch2d)[..., None] * rgb[None, None, :]
            return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)

        # READ ALL PYRAMID LEVELS
        image = self.load_image()  # dict[level] -> array

        sample_level = min(image.keys())
        arr0 = image[sample_level]
        if arr0.ndim < 3 or arr0.shape[2] < 1:
            console.print("No multiple channels found in image; fluorescence separation not applicable.", style="error")
            return self.metadata
        n_ch = int(arr0.shape[2])

        stain_dict = getattr(self.metadata[0], "stains", {}) or {}

        bg_values = None
        if self.remove_background:
            H0, W0 = arr0.shape[0], arr0.shape[1]
            stride = max(int(np.ceil(max(H0, W0) / MAX_BG_SIDE)), 1)
            A0_sample = np.asarray(arr0[::stride, ::stride, :])
            A_bg = A0_sample.astype(np.float32, copy=False)
            bg_values = np.array(
                [_estimate_channel_bg(A_bg[:, :, ch]) for ch in range(n_ch)],
                dtype=np.float32,
            )

        # preview (downsampled only)
        if self.preview:
            try:
                # choose a small level for preview if pyramidal
                preview_level = max(image.keys())
                arr_prev = image[preview_level]
                Hprev, Wprev = arr_prev.shape[0], arr_prev.shape[1]
                stride = max(int(np.ceil(max(Hprev, Wprev) / MAX_PREVIEW_SIDE)), 1)
                Aprev = np.asarray(arr_prev[::stride, ::stride, :])

                # Original panel
                if n_ch == 3:
                    # true color using the image's own channels
                    rgb = np.stack([_stretch99(Aprev[..., i]) for i in range(3)], axis=-1)
                    original_panel = (rgb * 255.0).astype(np.uint8)
                else:
                    # grayscale maximum projection for "original"
                    mip = np.max(Aprev.astype(np.float32), axis=2)
                    mip = (_stretch99(mip) * 255.0).astype(np.uint8)
                    original_panel = np.stack([mip, mip, mip], axis=-1)

                # Per-channel colored panels
                show_idx = list(range(min(n_ch, 6)))
                colored = []
                for ch in show_idx:
                    # use image's own colors if C==3 else palette
                    if n_ch == 3:
                        rgb_vec = np.zeros(3, dtype=np.float32)
                        rgb_vec[ch] = 1.0
                    else:
                        rgb_vec = PALETTE[ch % len(PALETTE)]
                    colored.append(_colorize(Aprev[..., ch], rgb_vec))

                # Layout: Original + colored channels in one row
                total = 1 + len(colored)
                fig, axes = plt.subplots(1, total, figsize=(3 * total, 3))
                axes = np.atleast_1d(axes).ravel()

                axes[0].imshow(original_panel)
                axes[0].set_title("Original")
                axes[0].axis("off")

                for i, img_rgb in enumerate(colored, start=1):
                    axes[i].imshow(img_rgb)
                    ch_name = stain_dict.get(show_idx[i - 1], f"Channel{show_idx[i - 1]}")
                    axes[i].set_title(ch_name)
                    axes[i].axis("off")

                plt.tight_layout()
                plt.show()

            except Exception as e:
                console.print(f"Fluorescence preview failed: {e}", style="error")

        # confirm before allocating full-res outputs
        if self.confirm:
            apply_full = Confirm.ask(f"Save {n_ch} separated channel images?", default=True, console=console)
            if not apply_full:
                console.print("Fluorescence channel separation aborted by user.", style="warning")
                del image  # Memory Management
                return self.metadata

        # prepare output paths and temporary memmap storage
        base_path = Path(self.metadata[0].path_storage)
        base_stem = base_path.name[:-len(''.join(base_path.suffixes))] if base_path.suffixes else base_path.stem
        dest_dir = Path(OUTPUT_PATH) / self.metadata[0].uid
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = dest_dir / "_tmp_mm"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # build full-resolution separated levels (disk backed, no downsampling)
        separated_levels = {ch: {} for ch in range(n_ch)}
        for level, arr in image.items():
            if arr.ndim != 3 or arr.shape[2] != n_ch:
                console.print(
                    f"Unexpected image format at level {level}, skipping.",
                    style="warning",
                )
                continue

            H, W = arr.shape[0], arr.shape[1]
            dtype = arr.dtype
            is_int = np.issubdtype(dtype, np.integer)
            if is_int:
                info = np.iinfo(dtype)

            # allocate one memmap per channel for this level
            mmaps = []
            for ch in range(n_ch):
                mm_path = tmp_dir / f"{base_stem}_L{level}_ch{ch}.dat"
                mm = np.memmap(mm_path, mode="w+", dtype=dtype, shape=(H, W))
                separated_levels[ch][level] = mm
                mmaps.append(mm)

            need_bg = self.remove_background and (bg_values is not None)

            # process this level tile by tile
            for ys, xs in _iter_tiles(arr, tile_size=2048):
                tile = np.asarray(arr[ys, xs, :], dtype=np.float32)

                if need_bg:
                    # subtract background from all channels at once
                    tile -= bg_values[None, None, :]
                    # clamp at 0 in place
                    np.maximum(tile, 0.0, out=tile)
                    if is_int:
                        # clamp to dtype max once for all channels
                        np.clip(tile, 0.0, float(info.max), out=tile)

                # write to channel-specific memmaps
                for ch, mm in enumerate(mmaps):
                    mm[ys, xs] = tile[:, :, ch].astype(dtype)

        # save per channel (grayscale intensity)
        output_metadata = []

        for ch in range(n_ch):
            # per-channel naming plus metadata
            stain_name = stain_dict.get(ch, f"ch{ch}")
            out_name = f"{base_stem}_{stain_name.replace(' ', '_')}.tiff"
            out_path = dest_dir / out_name

            new_meta = Metadata(
                path_original=self.metadata[0].path_original,
                path_storage=out_path,
                image_type=self.metadata[0].image_type or "fluorescence",
                uid=f"{self.metadata[0].uid}-ch{ch}",
            )
            new_meta.set_stains({0: stain_name})
            new_meta.save(dest_dir)
            # Pass the full level to array dict so save_tif writes a tiled, pyramidal TIFF
            save_tif(separated_levels[ch], out_path, metadata=new_meta)

            console.print(f"Saved channel {ch} -> {out_path}", style="success")
            output_metadata.append(new_meta)

        del image, separated_levels  # Memory Management

        return output_metadata


if __name__ == "__main__":
    from slidekick import DATA_PATH
    from slidekick.io import read_wsi
    """
    # Brightfield example
    image_path_brightfield = DATA_PATH / "reg" / "HE1.ome.tif"
    metadata_brightfield = Metadata(path_original=image_path_brightfield, path_storage=image_path_brightfield)

    # Check loaded img
    img, _ = read_wsi(metadata_brightfield.path_storage)
    print(img)

    bright = StainSeparator(metadata=metadata_brightfield, mode="brightfield", confirm=True, preview=True)
    metadatas_brightfield = bright.apply()

    # Check saved img
    img, _ = read_wsi(metadatas_brightfield[0].path_storage)
    print(img)
    """
    # Fluorescence example
    image_path_fluorescence = DATA_PATH / "reg" / "GS_CYP1A2.czi"
    metadata_fluorescence = Metadata(path_original=image_path_fluorescence, path_storage=image_path_fluorescence)

    # Check loaded img
    img, _ = read_wsi(metadata_fluorescence.path_storage)
    print(img)

    fluor = StainSeparator(metadata=metadata_fluorescence, mode="fluorescence", confirm=True, preview=True)
    metadatas_fluorescence = fluor.apply()

    # Check saved img
    img, _ = read_wsi(metadatas_fluorescence[0].path_storage)
    print(img)
