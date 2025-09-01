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

class StainSeparator(BaseOperator):

    def __init__(self,
                 metadata,
                 channel_selection=None,
                 mode: str = None,
                 stain_profile: str = None,
                 custom_matrix: np.ndarray = None,
                 confirm: bool = True,
                 preview: bool = True):
        """
        - mode: "brightfield" or "fluorescence" (auto if None)
        - stain_profile: e.g. "H&E", "H-DAB" (brightfield only)
        - custom_matrix: optional 3xN stain OD matrix (brightfield)
        """
        self.mode = mode
        self.stain_profile = stain_profile
        self.custom_matrix = custom_matrix
        self.confirm = confirm
        self.preview = preview
        super().__init__(metadata, channel_selection=None)

    def apply(self):
        """Execute stain separation on the input metadata."""
        mode = self.mode
        if mode is None:
            try:
                img_example = self.load_image()
                sample_level = next(iter(img_example.keys()))
                arr = _to_numpy(img_example[sample_level])
                if arr.ndim == 3 and arr.shape[2] == 3 and \
                   str(getattr(self.metadata[0], "image_type", "")).lower() in {"brightfield", "immunohistochemistry"}:
                    mode = "brightfield"
                elif arr.ndim == 3 and arr.shape[2] > 3:
                    mode = "fluorescence"
                else:
                    itype = str(getattr(self.metadata[0], "image_type", "")).lower()
                    mode = "fluorescence" if "fluorescence" in itype or "fluor" in itype else "brightfield"
            except Exception as e:
                console.print(f"Could not determine image modality automatically: {e}", style="warning")
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

        # also keep a colored preview per stain at preview level
        colored_preview = {}

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
                # Build grayscale per stain: I = 10^(-OD_stain)
                for j in range(C.shape[2]):
                    ODj = C[:, :, j]
                    Ij = np.power(10.0, -ODj)
                    out = (np.clip(Ij, 0.0, 1.0) * 255.0).astype(np.uint8)
                    separated_levels[j][level] = out
                # Build colored previews at this level once
                if level not in colored_preview:
                    colored_preview[level] = []
                    for j in range(C.shape[2]):
                        if stain_names[j].lower() == "residual":
                            continue
                        ODj = C[:, :, j]
                        col = M[:, j]  # (R,G,B)
                        OD_rgb = ODj[..., None] * col[None, None, :]
                        I_rgb = np.power(10.0, -np.clip(OD_rgb, 0.0, None))
                        rgb = (np.clip(I_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                        colored_preview[level].append(rgb)
            else:
                console.print(f"Level {level} image is not 3-channel, skipping level.", style="warning")

        # Preview (color)
        if self.preview:
            try:
                preview_level = _first_available_level(separated_levels)
                if preview_level is not None:
                    orig = _to_numpy(image.get(preview_level, next(iter(image.values()))))
                    color_panels = colored_preview.get(preview_level, [])
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
            # Passing a level→array dict => save_tif writes tiled, pyramidal TIFF
            save_tif(separated_levels[j], out_path, metadata=new_meta)

            console.print(f"Saved [{stain_name}] to {out_path}", style="success")
            output_metadata.append(new_meta)
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

        sample_level = min(image.keys())
        A0 = _to_numpy(image[sample_level])
        if A0.ndim < 3 or A0.shape[2] < 1:
            console.print("No multiple channels found in image; fluorescence separation not applicable.", style="error")
            return self.metadata
        n_ch = int(A0.shape[2])

        stain_dict = getattr(self.metadata[0], "stains", {}) or {}

        # preview (downsampled only)
        if self.preview:
            try:
                # choose a small level for preview if pyramidal
                preview_level = max(image.keys())
                Aprev = _to_numpy(image[preview_level])  # Y×X×C

                # cap preview longest side
                stride = max(int(np.ceil(max(Aprev.shape[0], Aprev.shape[1]) / MAX_PREVIEW_SIDE)), 1)
                if stride > 1:
                    Aprev = Aprev[::stride, ::stride, :]

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
                return self.metadata

        # build full-resolution separated levels (no downsampling)
        separated_levels = {ch: {} for ch in range(n_ch)}
        for level, arr in image.items():
            # keep ALL pyramid levels; each ch gets a level→2D array map
            A = _to_numpy(arr)  # Y×X×C
            if A.ndim != 3 or A.shape[2] != n_ch:
                console.print(f"Unexpected image format at level {level}, skipping.", style="warning")
                continue
            for ch in range(n_ch):
                separated_levels[ch][level] = np.asarray(A[:, :, ch])

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
            # Pass the full level→array dict so save_tif writes a tiled, pyramidal TIFF
            save_tif(separated_levels[ch], out_path, metadata=new_meta)

            console.print(f"Saved channel {ch} -> {out_path}", style="success")
            output_metadata.append(new_meta)

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
