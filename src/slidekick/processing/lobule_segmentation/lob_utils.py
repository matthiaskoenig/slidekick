import numpy as np
import cv2
from skimage.filters import threshold_multiotsu
from skimage.morphology import closing, disk
from skimage.color import label2rgb
from typing import Tuple, Dict, List, Optional, Any, Sequence, Callable
from collections.abc import Mapping
from pathlib import Path
from PIL import Image


def multiotsu_split(gray: np.ndarray, classes: int = 3, blur_sigma: float = 1.5, report_path: Path = None):
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
    if report_path is not None:
        # Build single-channel mask with requested gray codes
        vis = np.zeros_like(g, dtype=np.uint8)
        # Map each semantic class to its grayscale value
        if np.any(lbl == idx_true_bg):
            vis[lbl == idx_true_bg] = 0  # black
        if np.any(lbl == idx_mic_bg):
            vis[lbl == idx_mic_bg] = 128  # mid gray
        if np.any(lbl == idx_tissue):
            vis[lbl == idx_tissue] = 255  # white

        # Create parent dir if needed; accept file or directory path
        rp = Path(report_path)
        if rp.is_dir():
            rp = rp / "multiotsu_mask.png"
        rp.parent.mkdir(parents=True, exist_ok=True)

        # Write as PNG; lossless and preserves 8-bit grayscale
        cv2.imwrite(str(rp), vis)


    return lbl.astype(np.int32), (idx_true_bg, idx_mic_bg, idx_tissue)


def detect_tissue_mask_multiotsu(gray: np.ndarray,
                                 morphological_radius: int = 5, report_path: Path = None):
    """
    Build boolean masks: tissue, microscope background, true background.
    Cleans with closing + small object/hole removal.
    """
    lbl, (i0, i1, i2) = multiotsu_split(gray, classes=3, blur_sigma=15, report_path=report_path)
    m_tis  = (lbl == i2)

    m_tis = closing(m_tis, disk(int(morphological_radius)))

    return m_tis.astype(bool)


def overlay_mask(image_stack: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay segmentation labels using skimage.color.label2rgb.
    - Treats label 0 as background.
    - Supports multi-class masks.
    """

    base = image_stack.mean(axis=2).astype(np.float32) / 255.0
    lbl = mask.astype(np.int32)

    over = label2rgb(lbl, image=base, bg_label=0, alpha=alpha, image_alpha=1.0)
    return (over * 255).astype(np.uint8)


def bool_mask_to_uint8(mask: np.ndarray, on_val: int = 255) -> np.ndarray:
    """Convert a boolean/0-1 mask to uint8, using `on_val` for True pixels."""
    if mask.dtype == np.uint8:
        m = mask
        if m.max() <= 1:
            return (m * np.uint8(on_val)).astype(np.uint8)
        return m
    return (mask.astype(np.uint8) * np.uint8(on_val)).astype(np.uint8)


def minmax_to_uint8(img: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to uint8 [0,255]. Robust to constant/NaN arrays."""
    x = img.astype(np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(img.shape, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def border_connected_mask(binary: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Return a boolean mask of connected components in `binary` that touch the image border.
    """
    b = binary.astype(bool)
    if not np.any(b):
        return np.zeros(b.shape, dtype=bool)

    num, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=int(connectivity))
    if num <= 1:
        return np.zeros(b.shape, dtype=bool)

    border_ids = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return np.zeros(b.shape, dtype=bool)
    return np.isin(lab, border_ids)


def holes_from_fg_mask(fg_pix_mask: np.ndarray, border_exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """
    holes = fill(largest_external_contour(fg)) - fg
    """
    fg = fg_pix_mask.astype(bool)
    fg_u8 = bool_mask_to_uint8(fg, 255)

    cnts, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg_u8)
    if cnts:
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    else:
        filled[:] = fg_u8

    holes = (filled > 0) & (~fg)
    if border_exclude is not None:
        holes &= ~border_exclude.astype(bool)
    return holes


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

    # Step 3: build pyramid by resizing base canvas to each level
    out = {}
    for lvl, (Hdst, Wdst) in orig_shapes.items():
        if lvl == base_level:
            out[lvl] = canvas_base.copy()
        else:
            out[lvl] = cv2.resize(
                canvas_base, (Wdst, Hdst), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
    return out


def pad_image(image_stack: np.ndarray, pad: int) -> np.ndarray:
    pad_width = ((pad, pad), (pad, pad), (0, 0))
    return np.pad(image_stack, pad_width, mode="constant", constant_values=0)


def downsample_to_max_side(img: np.ndarray, max_side: int = 2048) -> np.ndarray:
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

    arr = arr.squeeze()

    pil = Image.fromarray(arr)
    pil = pil.resize((new_w, new_h), resample=Image.LANCZOS)
    out = np.asarray(pil)

    return out


def downsample_stack_to_max_side(stack: np.ndarray, max_side: int = 2048) -> np.ndarray:
    """
    Downsample each channel of an (H, W, C) stack so max(H, W) == max_side,
    preserving aspect ratio.

    Intended for interactive napari previews (fast + responsive).
    """
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError(
            f"Expected an ndarray stack of shape (H, W, C), got {type(stack)} {getattr(stack, 'shape', None)}"
        )

    frames = [downsample_to_max_side(stack[..., c], max_side) for c in range(stack.shape[2])]
    return np.stack(frames, axis=-1)


def gray_for_cluster(cid: int, sorted_label_idx: np.ndarray, n_clusters: int) -> int:
    """
    Map a cluster id to a grayscale value based on semantic order:
      position 0 (PP) -> 85, middle (MID) -> 170, last (PV) -> 255.
    Background stays 0 outside this function (caller decides).
    """
    try:
        pos = int(np.where(sorted_label_idx == cid)[0][0])
    except Exception:
        # Fallback: mid tone if cid is not in sorted_label_idx
        pos = 1 if n_clusters > 2 else 0
    if pos == 0:
        n = 1
    elif pos == (n_clusters - 1):
        n = 3
    else:
        n = 2
    return int(round(n * 255 / 3))


def render_cluster_gray(cluster_map: np.ndarray,
                        sorted_label_idx: np.ndarray,
                        n_clusters: int) -> np.ndarray:
    """
    Render a per-pixel cluster map to a grayscale template using gray_for_cluster.
    Leaves any values <0 (e.g., background/unassigned) at 0.
    """
    h, w = cluster_map.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for cid in range(n_clusters):
        out[cluster_map == cid] = gray_for_cluster(cid, sorted_label_idx, n_clusters)
    return out


def nonlinear_channel_weighting(
    X: np.ndarray,
    channels_pp: Optional[List[int]],
    channels_pv: Optional[List[int]],
    pp_gamma: float = 0.70,
    pv_gamma: float = 0.85,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> np.ndarray:
    """
    Percentile-normalize then gamma-lift selected columns of X.
    channels_pp get pp_gamma, channels_pv get pv_gamma. Returns a copy.
    """
    Xo = X.astype(np.float32, copy=True)
    eps = 1e-6

    def _lift(cols: List[int], gamma: float):
        if not cols:
            return
        for c in cols:
            v = Xo[:, c]
            lo = np.percentile(v, low_pct)
            hi = np.percentile(v, high_pct)
            if (hi - lo) <= eps:
                continue
            vn = np.clip((v - lo) / (hi - lo + eps), 0.0, 1.0)
            Xo[:, c] = (vn ** float(gamma)) * (hi - lo) + lo

    _lift(list(channels_pp) if channels_pp is not None else [], pp_gamma)
    _lift(list(channels_pv) if channels_pv is not None else [], pv_gamma)
    return Xo


def to_base_full(contours_xy: List[np.ndarray],
                  pad_px: int,
                  bbox_base: Tuple[int, int, int, int],
                  proc_hw: Tuple[int, int],
                  roi_hw_base: Tuple[int, int]) -> List[np.ndarray]:
    # contours_xy: OpenCV (x,y) in padded-ROI coordinates
    Hproc, Wproc = proc_hw
    Hb, Wb = roi_hw_base
    min_r, min_c, _, _ = bbox_base
    sx = Wb / float(max(Wproc, 1))
    sy = Hb / float(max(Hproc, 1))
    out: List[np.ndarray] = []
    for cnt in contours_xy:
        pts = np.asarray(cnt, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = (pts[:, 0] - float(pad_px)) * sx + float(min_c)
        pts[:, 1] = (pts[:, 1] - float(pad_px)) * sy + float(min_r)
        out.append(pts.reshape(-1, 1, 2).astype(np.int32))
    return out


def rescale_full(contours_xy: List[np.ndarray],
                  Hsrc: int, Wsrc: int, Hdst: int, Wdst: int) -> List[np.ndarray]:
    # scale contours from a full-frame source size to a full-frame destination size
    if not contours_xy:
        return []
    sx = float(Wdst) / float(max(Wsrc, 1))
    sy = float(Hdst) / float(max(Hsrc, 1))
    out: List[np.ndarray] = []
    for cnt in contours_xy:
        pts = np.asarray(cnt, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        out.append(pts.reshape(-1, 1, 2).astype(np.int32))
    return out


def discover_pyramid_levels(multiscale: Any) -> List[int]:
    """
    Best-effort discovery of available pyramid levels for a stain/multiscale object.

    Supported inputs (best effort):
      - Mapping[int|str -> array] (e.g. dict-like pyramids)
      - Sequence[array] (list/tuple multiscales)
      - Single array-like (treated as level [0])

    If levels cannot be determined, this returns [0] (required fallback).
    """
    try:
        if isinstance(multiscale, Mapping):
            levels: List[int] = []
            for k in multiscale.keys():
                try:
                    levels.append(int(k))
                except Exception:
                    continue
            return sorted(set(levels)) if levels else [0]

        if isinstance(multiscale, (list, tuple)):
            return list(range(len(multiscale))) if len(multiscale) else [0]

        # zarr-like groups often provide .keys() without being a Mapping
        if hasattr(multiscale, "keys") and callable(multiscale.keys):
            levels = []
            for k in multiscale.keys():
                try:
                    levels.append(int(k))
                except Exception:
                    continue
            return sorted(set(levels)) if levels else [0]
    except Exception:
        pass

    # Unknown / single-scale object.
    return [0]


def common_pyramid_levels(multiscales: Sequence[Any]) -> List[int]:
    """
    Return sorted intersection of available pyramid levels across all stains.

    If the intersection is empty (rare; inconsistent pyramids), this falls back to [0].
    """
    common: Optional[set] = None
    for ms in multiscales:
        lvls = set(discover_pyramid_levels(ms))
        common = lvls if common is None else (common & lvls)

    out = sorted(common) if common else []
    return out if out else [0]


def choose_default_preview_level(common_levels: Sequence[int]) -> int:
    """
    Default preview level: 0 if available, otherwise the lowest common level.
    """
    lvls = sorted(int(x) for x in common_levels) if common_levels else [0]
    return 0 if 0 in lvls else lvls[0]


def load_level_from_multiscale(multiscale: Any, level: int) -> Any:
    """
    Load a specific pyramid level from a multiscale object.

    - Mapping: tries int(level) then str(level)
    - Sequence: uses indexing
    - Otherwise: returns the object itself (assumed single-scale)
    """
    lvl = int(level)

    if isinstance(multiscale, Mapping):
        if lvl in multiscale:
            return multiscale[lvl]
        if str(lvl) in multiscale:
            return multiscale[str(lvl)]
        # best-effort: sort numeric keys and index
        keys = []
        for k in multiscale.keys():
            try:
                keys.append(int(k))
            except Exception:
                continue
        keys = sorted(set(keys))
        if keys and lvl < len(keys):
            return multiscale[keys[lvl]]
        raise KeyError(f"Level {lvl} not present in multiscale mapping.")

    if isinstance(multiscale, (list, tuple)):
        return multiscale[lvl]

    # Single-scale array/dask/zarr array etc.
    return multiscale


def discover_pyramid_shapes(multiscale: Any) -> Dict[int, Tuple[int, int]]:
    """
    Best-effort discovery of (H, W) per pyramid level.

    If a level fails to load, it is skipped.
    """
    out: Dict[int, Tuple[int, int]] = {}
    for lvl in discover_pyramid_levels(multiscale):
        try:
            arr = np.asarray(load_level_from_multiscale(multiscale, lvl))
            if arr.ndim >= 2:
                out[int(lvl)] = (int(arr.shape[0]), int(arr.shape[1]))
        except Exception:
            continue
    return out


def add_napari_controls_dock(
    viewer: Any,
    controls_widget: Any,
    *,
    on_update: Optional[Callable[[], None]] = None,
    on_confirm: Optional[Callable[[], None]] = None,
    on_back: Optional[Callable[[], None]] = None,
    on_reset: Optional[Callable[[], None]] = None,
    update_text: str = "Preview / Update",
    confirm_text: str = "Confirm and continue",
    back_text: str = "Back",
    reset_text: str = "Reset to defaults",
    include_update: bool = True,
    include_confirm: bool = True,
    include_back: bool = False,
    include_reset: bool = False,
    dock_area: str = "right",
) -> Dict[str, Any]:
    """
    Unify napari preview UI wiring across the codebase.

    Notes on environment compatibility
    ---------------------------------
    Different napari/magicgui versions expose PushButton signals differently:
      - some versions use `changed`
      - others use `clicked`
      - some require wiring through the underlying Qt widget (`.native`)

    Critical robustness fix
    ----------------------
    Some Qt bindings (notably some PySide setups) can drop callbacks connected via
    anonymous lambdas if no Python reference is kept. We therefore create a
    wrapper function and store it on the button to prevent GC.

    Returns
    -------
    Dict[str, Any]
        Mapping of button names to PushButton widgets for optional external wiring.
    """
    from magicgui.widgets import Container, PushButton
    import warnings

    def _btn_text(btn: Any) -> str:
        txt = getattr(btn, "text", None)
        try:
            return str(txt()) if callable(txt) else str(txt)
        except Exception:
            return "<unknown>"

    def _wire_button(btn: Any, cb: Optional[Callable[[], None]]) -> None:
        """Best-effort connect for magicgui/Qt push buttons across versions (GC-safe)."""
        if cb is None:
            return

        def _invoke_cb(*_args: Any, **_kwargs: Any) -> None:
            try:
                cb()
            except Exception as e:
                warnings.warn(
                    f"add_napari_controls_dock: callback error for button '{_btn_text(btn)}': {e!r}",
                    RuntimeWarning,
                )

        # Keep a strong reference to avoid GC issues in some environments.
        try:
            setattr(btn, "_napari_controls_cb_ref", _invoke_cb)
        except Exception:
            pass

        # Prefer underlying Qt signal if available (most reliable across versions)
        native = getattr(btn, "native", None)
        if native is not None:
            for sig_name in ("clicked", "pressed", "released"):
                sig = getattr(native, sig_name, None)
                if sig is None:
                    continue
                try:
                    sig.connect(_invoke_cb)
                    return
                except Exception:
                    pass

        # Fall back to magicgui-level signals
        for sig_name in ("clicked", "changed"):
            sig = getattr(btn, sig_name, None)
            if sig is None:
                continue
            try:
                sig.connect(_invoke_cb)
                return
            except Exception:
                pass

        warnings.warn(
            f"add_napari_controls_dock: could not connect callback for button '{_btn_text(btn)}'.",
            RuntimeWarning,
        )

    widgets = [controls_widget]
    out: Dict[str, Any] = {}

    if include_update:
        update_btn = PushButton(text=str(update_text))
        _wire_button(update_btn, on_update)
        widgets.append(update_btn)
        out["update"] = update_btn

    if include_confirm:
        confirm_btn = PushButton(text=str(confirm_text))
        _wire_button(confirm_btn, on_confirm)
        widgets.append(confirm_btn)
        out["confirm"] = confirm_btn

    if include_back:
        back_btn = PushButton(text=str(back_text))
        _wire_button(back_btn, on_back)
        widgets.append(back_btn)
        out["back"] = back_btn

    if include_reset:
        reset_btn = PushButton(text=str(reset_text))
        _wire_button(reset_btn, on_reset)
        widgets.append(reset_btn)
        out["reset"] = reset_btn

    dock = Container(widgets=widgets, layout="vertical")
    viewer.window.add_dock_widget(dock, area=str(dock_area))
    return out


def preview_images_napari(
    images: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    *,
    require_confirm: bool = False,
    return_action: bool = False,
    include_confirm: bool = True,
    include_back: bool = False,
    confirm_text: str = "Confirm",
    back_text: str = "Back",
    dock_area: str = "right",
) -> Optional[str]:
    """
    Small helper to preview one or more images in napari.

    Images are downsampled (max side 2048 px) for responsiveness.

    New (optional) behavior:
      - Can show Confirm/Back buttons in a dock.
      - Can return an action string in {"confirm", "back", "closed"}.

    Parameters
    ----------
    require_confirm:
        If True, closing the viewer without pressing Confirm returns "closed".
        If False, closing the viewer maps to "confirm".
    return_action:
        If True, return one of {"confirm","back","closed"}.
        If False, preserve legacy behavior and return None.
    include_confirm / include_back:
        Toggle whether to show Confirm/Back buttons.
    confirm_text / back_text:
        Button labels.
    dock_area:
        Napari dock area; usually "right".
    """
    if not images:
        return ("confirm" if not require_confirm else "closed") if return_action else None

    import napari
    from magicgui.widgets import Label

    thumbs = [downsample_to_max_side(np.asarray(im), 2048) for im in images]
    viewer = napari.Viewer()

    for idx, im in enumerate(thumbs):
        name = str(titles[idx]) if titles and idx < len(titles) else f"Layer_{idx}"
        if im.ndim == 2:
            viewer.add_image(im, name=name, colormap="gray")
        elif im.ndim == 3 and im.shape[-1] in (3, 4):
            viewer.add_image(im, name=name, rgb=True)
        else:
            viewer.add_image(im, name=name)

    # Navigation result
    nav: Dict[str, Optional[str]] = {"action": None}  # "confirm" | "back" | None (closed)

    def on_confirm() -> None:
        nav["action"] = "confirm"
        viewer.close()

    def on_back() -> None:
        nav["action"] = "back"
        viewer.close()

    # Minimal "controls" widget so we can reuse the shared dock builder.
    # (Keeps UI consistent across previews.)
    controls_stub = Label(value="")

    if include_confirm or include_back:
        add_napari_controls_dock(
            viewer,
            controls_stub,
            on_confirm=on_confirm if include_confirm else None,
            on_back=on_back if include_back else None,
            include_update=False,
            include_confirm=bool(include_confirm),
            include_back=bool(include_back),
            include_reset=False,
            confirm_text=str(confirm_text),
            back_text=str(back_text),
            dock_area=str(dock_area),
        )

    napari.run()

    if not return_action:
        return None

    action = nav["action"]
    if action is None:
        action = "confirm" if not require_confirm else "closed"
    return action

