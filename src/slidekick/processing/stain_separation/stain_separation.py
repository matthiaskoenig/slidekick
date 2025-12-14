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
}
DEFAULT_STAIN_NAMES = {
    "H&E": ["Hematoxylin", "Eosin"],
    "H-DAB": ["Hematoxylin", "DAB"],
    "HED": ["Hematoxylin", "Eosin", "DAB"],
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


def _to_cyx_fluorescence(arr_np: np.ndarray) -> np.ndarray:
    """
    Normalize fluorescence arrays to (C, Y, X).

    Supports common layouts:
      - (C, Y, X)          [CYX]
      - (Y, X, C)          [YXC]
      - (T, C, Z, Y, X)    [TCZYX]
      - (C, Z, Y, X)       [CZYX]
      - (Z, C, Y, X)       [ZCYX]

    For 4D/5D we assume the last two axes are spatial (Y, X), keep the first
    plausible channel axis (small dimension <=64), and slice other leading axes at 0.
    """
    a = np.asarray(arr_np)

    if a.ndim == 2:
        return a[None, :, :]  # (1, Y, X)

    if a.ndim == 3:
        # Already CYX (channel-first)
        if a.shape[0] <= 64 and a.shape[1] > a.shape[0] and a.shape[2] > a.shape[0]:
            return a
        # YXC (channel-last)
        if a.shape[2] <= 64 and a.shape[0] > a.shape[2] and a.shape[1] > a.shape[2]:
            return np.moveaxis(a, -1, 0)
        # Fallback: smallest axis is probably channels
        cax = int(np.argmin(a.shape))
        return np.moveaxis(a, cax, 0)

    if a.ndim in (4, 5):
        lead_axes = list(range(a.ndim - 2))  # everything except Y,X

        cand = [i for i in lead_axes if 2 <= a.shape[i] <= 64]
        if not cand:
            cand = [i for i in lead_axes if a.shape[i] > 1]
        cax = cand[0] if cand else lead_axes[0]

        slicer = []
        for i in range(a.ndim):
            if i == cax:
                slicer.append(slice(None))  # keep C
            elif i in lead_axes:
                slicer.append(0)            # take first T/Z/...
            else:
                slicer.append(slice(None))  # keep Y,X
        b = a[tuple(slicer)]
        return _to_cyx_fluorescence(b)

    return _to_cyx_fluorescence(np.squeeze(a))


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
        """Execute stain separation on the input metadata.

        Mode detection is robust to CYX vs YXC and to 3-channel fluorescence:
        - Use metadata hint if available.
        - Use channel axis heuristics (small dimension up to 16).
        - Probe 3-channel uint8 data to decide if it is true RGB.
        """
        mode = self.mode
        if mode is None:
            try:
                img_example = self.load_image()
                sample_level = next(iter(img_example.keys()))
                arr0 = _to_numpy(img_example[sample_level])

                shape = getattr(arr0, "shape", None)
                ndim = arr0.ndim if shape is not None else 0

                # Hints from metadata
                itype = str(getattr(self.metadata[0], "image_type", "") or "").lower()
                hint_fluor = ("fluor" in itype) or ("fluorescence" in itype)
                hint_bright = any(
                    k in itype
                    for k in [
                        "brightfield",
                        "immunohistochemistry",
                        "ihc",
                        "h&e",
                        "he",
                        "dab",
                    ]
                )

                nchan = 1
                is_color_probe = False

                if ndim == 3:
                    s0, s1, s2 = arr0.shape

                    # Heuristic: channel axis is the smallest dimension (<=16)
                    if s0 <= 16 and s0 < s1 and s0 < s2:
                        ch_axis = 0  # C, Y, X
                    elif s2 <= 16 and s2 < s0 and s2 < s1:
                        ch_axis = 2  # Y, X, C
                    else:
                        # Fall back to channels-last (typical for RGB TIFFs)
                        ch_axis = 2

                    nchan = int(arr0.shape[ch_axis])

                    # For 3 channels, probe a small patch to see if it "looks RGB"
                    if nchan == 3:
                        if ch_axis == 0:
                            H, W = arr0.shape[1], arr0.shape[2]
                            ph = min(32, H)
                            pw = min(32, W)
                            probe = np.moveaxis(arr0[:, :ph, :pw], 0, 2)  # (C,H,W)->(H,W,C)
                        else:
                            H, W = arr0.shape[0], arr0.shape[1]
                            ph = min(32, H)
                            pw = min(32, W)
                            probe = arr0[:ph, :pw, :]

                        is_color_probe = (
                            probe.ndim == 3
                            and probe.shape[-1] == 3
                            and probe.dtype == np.uint8
                        )

                # Decision logic
                if hint_fluor and not (is_color_probe and nchan == 3 and hint_bright):
                    mode = "fluorescence"
                elif is_color_probe and nchan == 3:
                    mode = "brightfield"
                elif hint_bright and (nchan == 1 or ndim < 3):
                    mode = "brightfield"
                elif ndim == 3 and nchan > 1:
                    mode = "fluorescence"
                else:
                    mode = "brightfield"

            except Exception as e:
                console.print(
                    f"Could not determine image modality automatically: {e}",
                    style="warning",
                )
                mode = "brightfield"

        console.print(f"StainSeparator mode: {mode}", style="info")
        if mode == "brightfield":
            return self._apply_brightfield()
        else:
            return self._apply_fluorescence()


    # brightfield
    def _apply_brightfield(self):
        M, stain_names, profile = self._resolve_brightfield_profile()
        # normalize columns and add residual if needed
        M = np.array([c / (np.linalg.norm(c) + 1e-8) for c in M.T], dtype=float).T
        if M.shape[1] == 2:
            v1, v2 = M[:, 0], M[:, 1]
            v3 = np.cross(v1, v2)
            n = np.linalg.norm(v3)
            if n < 1e-6:
                raise ValueError(f"Provided stain vectors are nearly collinear for profile {profile}.")
            v3 /= n
            M = np.column_stack([M, v3])
            stain_names = stain_names + ["Residual"]

        # READ ALL PYRAMID LEVELS
        image = self.load_image()   # dict[level] -> array
        separated_levels = [dict() for _ in range(M.shape[1])]

        for level, img in image.items():
            img_np = _to_numpy(img)
            if img_np.ndim == 3 and img_np.shape[2] >= 3:
                arr01 = _normalize_image_01(img_np)
                # Optical density: -log10(I)
                OD = -np.log10(arr01)
                H, W, _ = OD.shape
                OD_2d = OD.reshape(-1, 3)
                # Invert deconvolution matrix
                try:
                    M_inv = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    M_inv = np.linalg.pinv(M)
                C = OD_2d.dot(M_inv.T).reshape(H, W, -1)
                # Build RGB per stain (keep brightfield separated outputs as color)
                for j in range(C.shape[2]):
                    if stain_names[j].lower() == "residual":
                        continue
                    ODj = C[:, :, j]
                    col = M[:, j]  # (R,G,B)
                    OD_rgb = ODj[..., None] * col[None, None, :]
                    I_rgb = np.power(10.0, -np.clip(OD_rgb, 0.0, None))
                    separated_levels[j][level] = (np.clip(I_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

            else:
                console.print(f"Level {level} image is not 3-channel, skipping level.", style="warning")

        # Preview (color)
        if self.preview:
            try:
                preview_level = _first_available_level(separated_levels)
                if preview_level is not None:
                    orig = _to_numpy(image.get(preview_level, next(iter(image.values()))))
                    color_panels = [
                        separated_levels[j][preview_level]
                        for j, n in enumerate(stain_names)
                        if n.lower() != "residual" and preview_level in separated_levels[j]
                    ]
                    names_for_preview = [n for n in stain_names if n.lower() != "residual"]
                    total = 1 + len(color_panels)
                    cols = min(total, 3)
                    rows = int(np.ceil(total / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
                    axes = np.array(axes).reshape(-1)
                    axes[0].imshow(orig if orig.shape[-1] == 3 else orig[..., :3])
                    axes[0].set_title("Original"); axes[0].axis('off')
                    for k, panel in enumerate(color_panels, start=1):
                        axes[k].imshow(panel)
                        axes[k].set_title(names_for_preview[k-1]); axes[k].axis('off')
                    for ax in axes[1+len(color_panels):]:
                        ax.axis('off')
                    plt.tight_layout(); plt.show()
            except Exception as e:
                console.print(f"Preview generation failed: {e}", style="error")

        # Confirm
        if self.confirm:
            apply_full = Confirm.ask("Apply stain separation to full resolution image(s)?", default=True, console=console)
            if not apply_full:
                console.print("Stain separation aborted by user.", style="warning")
                del image, separated_levels  # Memory Management
                return self.metadata

        # Save outputs and create metadata per stain
        output_metadata = []
        base_path = Path(self.metadata[0].path_storage)
        # safe multi-suffix trimming (e.g., .ome.tif)
        base_stem = base_path.name[:-len(''.join(base_path.suffixes))] if base_path.suffixes else base_path.stem

        dest_dir = Path(OUTPUT_PATH) / self.metadata[0].uid
        dest_dir.mkdir(parents=True, exist_ok=True)

        for j, stain_name in enumerate(stain_names):
            if stain_name.lower() == "residual":
                continue

            out_name = f"{base_stem}_{stain_name.replace(' ', '_')}.tiff"
            out_path = dest_dir / out_name

            new_meta = Metadata(
                path_original=self.metadata[0].path_original,
                path_storage=out_path,
                image_type=self.metadata[0].image_type,
                uid=f"{self.metadata[0].uid}-{stain_name.replace(' ', '_')}"
            )
            new_meta.set_stains({0: stain_name})
            new_meta.save(dest_dir)  # save metadata into the same folder

            # Minimal OME metadata for an RGB YXS pyramid
            ome_meta = {
                "axes": "YXS",  # per-level arrays are H×W×3 (RGB samples)
                "Channel": {"Name": [stain_name]},  # one logical channel, RGB stored as samples
            }

            # Passing a level→array dict + ome_metadata => pyramidal OME-TIFF
            # `metadata=new_meta` is kept for API compatibility but ignored by save_tif
            # when it is not a dict.
            save_tif(separated_levels[j], out_path, metadata=new_meta, ome_metadata=ome_meta)

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

        # Probe a small pyramid level to avoid reading full-res just to infer channel axes
        probe_level = max(image.keys())
        A0 = _to_cyx_fluorescence(_to_numpy(image[probe_level]))

        if A0.ndim != 3 or A0.shape[0] < 1:
            console.print("No multiple channels found in image; fluorescence separation not applicable.", style="error")
            return self.metadata
        n_ch = int(A0.shape[0])

        stain_dict = getattr(self.metadata[0], "stains", {}) or {}

        bg_values = None
        if self.remove_background:
            A_bg = A0.astype(np.float32, copy=False)  # (C, Y, X)
            bg_values = []
            for ch in range(n_ch):
                bg = _estimate_channel_bg(A_bg[ch, :, :])
                bg_values.append(bg)
            bg_values = np.array(bg_values, dtype=np.float32)

        # preview (downsampled only)
        if self.preview:
            try:
                # choose a small level for preview if pyramidal
                preview_level = max(image.keys())
                Aprev = _to_cyx_fluorescence(_to_numpy(image[preview_level]))  # (C, Y, X)

                # cap preview longest side (over Y,X)
                stride = max(int(np.ceil(max(Aprev.shape[1], Aprev.shape[2]) / MAX_PREVIEW_SIDE)), 1)
                if stride > 1:
                    Aprev = Aprev[:, ::stride, ::stride]

                # Original panel
                if n_ch == 3:
                    # true color using the image's own channels
                    rgb = np.stack([_stretch99(Aprev[i, :, :]) for i in range(3)], axis=-1)  # (Y, X, 3)
                    original_panel = (rgb * 255.0).astype(np.uint8)
                else:
                    # grayscale maximum projection for "original"
                    mip = np.max(Aprev.astype(np.float32), axis=0)  # (Y, X)
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
                    colored.append(_colorize(Aprev[ch, :, :], rgb_vec))

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

        # build full-resolution separated levels (no downsampling)
        separated_levels = {ch: {} for ch in range(n_ch)}
        for level, arr in image.items():
            A = _to_cyx_fluorescence(_to_numpy(arr))  # (C, Y, X)
            if A.ndim != 3 or A.shape[0] != n_ch:
                console.print(f"Unexpected image format at level {level}, skipping.", style="warning")
                continue

            # Remove background
            if not self.remove_background or bg_values is None:
                for ch in range(n_ch):
                    separated_levels[ch][level] = np.asarray(A[ch, :, :])
                continue

            A_float = A.astype(np.float32, copy=False)

            for ch in range(n_ch):
                ch_arr = A_float[ch, :, :] - bg_values[ch]
                ch_arr[ch_arr < 0] = 0

                if np.issubdtype(A.dtype, np.integer):
                    info = np.iinfo(A.dtype)
                    ch_out = np.clip(ch_arr, 0, info.max).astype(A.dtype)
                else:
                    ch_out = ch_arr.astype(A.dtype)

                separated_levels[ch][level] = ch_out

        # save per channel (grayscale intensity)
        output_metadata = []
        base_path = Path(self.metadata[0].path_storage)
        # safe multi-suffix trimming (e.g., .ome.tif)
        base_stem = base_path.name[:-len(''.join(base_path.suffixes))] if base_path.suffixes else base_path.stem
        dest_dir = Path(OUTPUT_PATH) / self.metadata[0].uid
        dest_dir.mkdir(parents=True, exist_ok=True)

        for ch in range(n_ch):
            # per-channel naming + metadata
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

            # Minimal OME metadata for a single-channel YX pyramid
            ome_meta = {
                "axes": "YX",  # per-level arrays are H×W
                "Channel": {"Name": [stain_name]},  # OME <Channel Name=...>
            }

            # Pass the full level→array dict so save_tif writes a tiled, pyramidal OME-TIFF
            # `metadata=new_meta` is kept for API compatibility but ignored as it is not a dict.
            save_tif(separated_levels[ch], out_path, metadata=new_meta, ome_metadata=ome_meta)

            console.print(f"Saved channel {ch} -> {out_path}", style="success")
            output_metadata.append(new_meta)

        del image, separated_levels  # Memory Management

        return output_metadata


if __name__ == "__main__":
    from slidekick import DATA_PATH
    from slidekick.io import read_wsi

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
