import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label as ndi_label
from skimage.filters import threshold_multiotsu, threshold_otsu, threshold_triangle
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.color import label2rgb
from typing import Tuple, Dict, List, Optional, Any, Sequence
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
            rp = rp / "tissue_map.png"
        rp.parent.mkdir(parents=True, exist_ok=True)

        # Write as PNG; lossless and preserves 8-bit grayscale
        cv2.imwrite(str(rp), vis)


    return lbl.astype(np.int32), (idx_true_bg, idx_mic_bg, idx_tissue), thr


# Fraction of pixels in the middle intensity class below which the histogram
# is considered bimodal (true BG + tissue only, no microscopy artefact band).
_MIC_BG_THRESHOLD = 0.05

# Minimum relative gap between the two 3-class Otsu thresholds (fraction of
# the intensity range). When the thresholds are closer than this, the
# histogram is effectively bimodal even if mic_bg_frac >= _MIC_BG_THRESHOLD.
_MIN_THRESHOLD_GAP = 0.10


def _is_bimodal(lbl: np.ndarray, idx_mid: int, thr_low: float, thr_high: float) -> bool:
    """Decide whether a 3-class Otsu split really reflects 2 or 3 modes.

    Two independent heuristics, either one triggering means bimodal:
      1. The middle class contains < 5% of pixels (it's a forced artefact).
      2. The two Otsu thresholds are within 10 % of the 0-255 range of each
         other, meaning the algorithm couldn't find a real valley between
         them.
    """
    mic_bg_frac = float(np.sum(lbl == idx_mid)) / max(lbl.size, 1)
    if mic_bg_frac < _MIC_BG_THRESHOLD:
        return True
    gap = abs(float(thr_high) - float(thr_low)) / 255.0
    if gap < _MIN_THRESHOLD_GAP:
        return True
    return False


def _morphological_cleanup(m_tis: np.ndarray, dim: int, area: int,
                           median_frac: float, hole_area_frac: float,
                           min_obj_frac: float) -> np.ndarray:
    """Resolution-invariant morphological cleanup for tissue masks.

    Optional median filter smooths salt-and-pepper noise at the boundary
    without filling concavities or eroding thin peninsulas. Skipped
    when *median_frac* is 0.
    """
    # 1) Optional median filter for boundary smoothing.
    if median_frac > 0:
        ksize = max(int(median_frac * dim), 3)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        m_tis = cv2.medianBlur(m_tis.astype(np.uint8) * 255, ksize) > 127

    # 2) Fill all fully enclosed holes (dim zones inside tissue).
    m_tis = binary_fill_holes(m_tis)

    # 3) Remove small holes that may touch the image border.
    hole_area = max(int(hole_area_frac * area), 1)
    m_tis = remove_small_holes(m_tis, area_threshold=hole_area)

    # 4) Drop tiny spurious foreground specks.
    min_obj = max(int(min_obj_frac * area), 1)
    m_tis = remove_small_objects(m_tis, min_size=min_obj)

    return m_tis.astype(bool)


def detect_tissue_mask_multiotsu(gray: np.ndarray,
                                 morphological_radius: int = 5,
                                 auto: bool = True,
                                 report_path: Path = None,
                                 blur_frac: float = 0.007,
                                 close_frac: float = 0.01,
                                 hole_area_frac: float = 0.05,
                                 min_obj_frac: float = 0.01,
                                 outline_smooth_frac: float = 0.0):
    """Build a boolean tissue mask via multi-Otsu thresholding.

    1. 3-class Otsu -> labels for true BG, mic BG, tissue.
    2. Bimodal check -> fall back to 2-class Otsu if only 2 real modes.
    3. Trimodal: tissue = brightest class; bimodal: tissue = above threshold.
    4. Border-connected BG removal fills interior vessel holes.
    5. Morphological cleanup (fill holes, remove specks).

    All spatial parameters are fractions of image dimensions for
    resolution invariance.
    """
    H, W = gray.shape[:2]
    dim = min(H, W)
    area = H * W

    # Step 1: resolution-invariant blur
    sigma = max(int(blur_frac * dim), 1)

    # Step 2: 3-class Otsu via multiotsu_split (also saves debug image)
    lbl3, (idx_bg, idx_mid, idx_tis), thr3 = multiotsu_split(
        gray, classes=3, blur_sigma=sigma, report_path=report_path)

    # Step 3: bimodal/trimodal decision -> initial binary mask
    means = [float(gray[lbl3 == i].mean()) if np.any(lbl3 == i)
             else 0.0 for i in range(3)]

    if auto and _is_bimodal(lbl3, idx_mid, float(thr3[0]), float(thr3[1])):
        g_blur = cv2.GaussianBlur(gray.astype(np.uint8), (0, 0), sigma)
        try:
            thr2 = threshold_otsu(g_blur)
        except Exception:
            thr2 = float(np.mean(gray)) - 1.0
        m_tis = (gray > thr2)
    elif means[idx_bg] > 10:
        # Darkest class contains mic BG (mean well above 0) -> Otsu lumped
        # true BG + mic BG together and split tissue into mid/top.
        # Tissue = mid + top class (everything except darkest).
        m_tis = (lbl3 != idx_bg)
    else:
        # Darkest class is pure true BG (near 0) -> mid class is mic BG.
        # Tissue = top class only.
        m_tis = (lbl3 == idx_tis)

    # Step 4: border-connected BG removal
    #   Small closing first to bridge vessel gaps at the tissue boundary.
    #   Without this, boundary vessels create paths from tissue interior
    #   to external background, causing undersegmentation.
    close_r = max(int(close_frac * dim / 2), 1)
    if close_r >= 2:
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        m_tis = cv2.morphologyEx(
            m_tis.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)

    #   Label all non-tissue components; only those touching the image
    #   border are real background. Interior holes (vessels) survive.
    bg_labeled, n = ndi_label(~m_tis)
    if n > 0:
        border_ids = set()
        border_ids.update(bg_labeled[0, :].ravel())
        border_ids.update(bg_labeled[-1, :].ravel())
        border_ids.update(bg_labeled[:, 0].ravel())
        border_ids.update(bg_labeled[:, -1].ravel())
        border_ids.discard(0)
        m_tis = ~np.isin(bg_labeled, list(border_ids))

    # Step 5: morphological cleanup. outline_smooth_frac > 0 enables a median
    # filter on the mask (fraction of the smaller dimension) that rounds off the
    # jagged tissue outline without eroding concavities; 0 keeps the raw outline.
    return _morphological_cleanup(m_tis, dim, area, outline_smooth_frac,
                                  hole_area_frac, min_obj_frac)


# A 3-class mean at or below this raw value counts as a near-black background.
_NEAR_BLACK = 10.0
# Minimum fraction of the frame that must be near-black for a stain to be
# considered tissue-positive (there is a real dark background to threshold off).
_MIN_BLACK_FRAC = 0.05


def is_tissue_positive(gray: np.ndarray) -> bool:
    """True when the stain is mainly positive inside tissue over a near-black bg.

    Decided from the 3-class Otsu mode structure (the same split the multi-Otsu
    detector runs, so the two agree): a tissue-positive stain has its darkest
    mode near black and a middle mode that belongs to the tissue continuum, i.e.
    closer to the bright mode than to black. A slide with a microscopy-background
    band instead has a mid mode sitting in the valley between black and tissue
    (closer to black), or a darkest mode that is itself a bright band. That mode
    test is robust to broad, skewed tissue intensity, which a simple threshold or
    a bimodal-fraction test mistakes for a microscopy band.

    Tissue-positive is exactly where a global Triangle threshold with minimal
    cleanup follows the tissue edge, while the multi-Otsu path (which removes the
    bright band at the cost of rounding the outline) is needed otherwise.
    """
    g = gray.astype(np.uint8)
    if float(np.mean(gray <= _NEAR_BLACK)) < _MIN_BLACK_FRAC:
        return False  # no real dark background to threshold off
    sigma = max(int(0.007 * min(g.shape[:2])), 1)
    lbl3, (idx_bg, idx_mid, idx_tis), _ = multiotsu_split(
        g, classes=3, blur_sigma=sigma, report_path=None)

    def _cmean(i: int) -> float:
        return float(gray[lbl3 == i].mean()) if np.any(lbl3 == i) else 0.0

    m_bg, m_mid, m_top = _cmean(idx_bg), _cmean(idx_mid), _cmean(idx_tis)
    if m_bg > _NEAR_BLACK:
        return False  # darkest mode is itself a bright band -> microscopy bg
    # Middle mode belongs to the tissue continuum (nearer the bright mode) rather
    # than being a dim background band (nearer black).
    return (m_top - m_mid) < (m_mid - m_bg)


def detect_tissue_mask_triangle(gray: np.ndarray,
                                method: str = "triangle",
                                blur_frac: float = 0.002,
                                hole_area_frac: float = 0.05,
                                min_obj_frac: float = 0.01,
                                outline_smooth_frac: float = 0.0,
                                keep_largest: bool = True,
                                report_path: Path = None) -> np.ndarray:
    """Single-threshold tissue mask for tissue-positive stains.

    For a stain that is bright inside tissue over a near-black background, a
    global Triangle (``method="triangle"``) or Otsu (``method="otsu"``)
    threshold with only minimal blur and light cleanup follows the true tissue
    boundary far better than the multi-Otsu + ellipse-closing path, which
    rounds the outline into circular artefacts. No heavy closing is applied.

    Spatial parameters are fractions of image dimensions for resolution
    invariance.
    """
    H, W = gray.shape[:2]
    dim = min(H, W)
    area = H * W

    # Minimal blur only: just enough to steady the threshold, not to round
    # the outline. blur_frac defaults well below the multi-Otsu 0.007.
    sigma = max(int(blur_frac * dim), 0)
    g = gray.astype(np.uint8)
    if sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigma)
    try:
        thr = threshold_otsu(g) if method == "otsu" else threshold_triangle(g)
    except Exception:
        thr = float(np.mean(gray)) - 1.0
    m_tis = gray > thr

    # Border-connected background removal: keep interior holes (vessels) as
    # tissue, drop only background that reaches the image border. No closing,
    # so the outline is not rounded.
    m_tis = ~border_connected_mask(~m_tis)

    # Light cleanup: fill holes, drop specks, optional outline median.
    m_tis = _morphological_cleanup(m_tis, dim, area, outline_smooth_frac,
                                   hole_area_frac, min_obj_frac)

    # Keep the single largest tissue component.
    if keep_largest and np.any(m_tis):
        lab, n = ndi_label(m_tis)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m_tis = lab == int(np.argmax(sizes))

    if report_path is not None:
        rp = Path(report_path)
        if rp.is_dir():
            rp = rp / "tissue_map.png"
        rp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rp), (m_tis.astype(np.uint8) * 255))

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


def edge_lobule_labels(labels: np.ndarray,
                       tissue_footprint: np.ndarray,
                       margin_px: float,
                       majority_frac: float = 0.5,
                       min_area: int = 10) -> set:
    """Label ids of lobules that are incomplete at the tissue section edge.

    A lobule is flagged when it both *touches* the outer tissue boundary and has
    the *majority* of its area close to that boundary, i.e. a cut, incomplete
    lobule rather than a full one that merely kisses the edge. This mirrors how a
    histologist ignores edge lobules.

    Parameters
    ----------
    labels : np.ndarray
        Integer lobule label map (0 = background, >0 = lobule id).
    tissue_footprint : np.ndarray
        Boolean silhouette of the tissue (holes filled). Its outer contour is the
        section boundary a lobule can be cut by.
    margin_px : float
        Distance from the boundary within which a pixel counts as "close to the
        edge". Roughly one lobule radius: a full disc tangent to the boundary sits
        at the 50% threshold, so a half-cut lobule is flagged and a complete one is
        not.
    majority_frac : float
        A lobule is flagged when more than this fraction of its area lies within
        *margin_px* of the boundary.
    min_area : int
        Ignore labels smaller than this (noise).

    Returns
    -------
    set of int
        Label ids to exclude.
    """
    foot = tissue_footprint.astype(bool)
    if labels.size == 0 or int(labels.max()) == 0:
        return set()
    # Distance of every pixel from the exterior (0 on the tissue boundary).
    dist = distance_transform_edt(foot).astype(np.float32)

    # Aggregate per label in a single pass with bincount rather than looping and
    # scanning the whole frame once per label (O(pixels) instead of O(labels x
    # pixels)). `touch` counts pixels on the boundary (dist <= 1) and `near`
    # counts pixels within the margin; a label is an edge lobule when it touches
    # the boundary and the majority of its area is near it.
    lab = labels.ravel().astype(np.intp)
    d = dist.ravel()
    n = int(labels.max()) + 1
    area = np.bincount(lab, minlength=n).astype(np.float64)
    near = np.bincount(lab, weights=(d < float(margin_px)).astype(np.float64),
                       minlength=n)
    touch = np.bincount(lab, weights=(d <= 1.0).astype(np.float64), minlength=n)

    excluded = set()
    for lid in range(1, n):
        a = area[lid]
        if a < min_area:
            continue
        if touch[lid] > 0 and (near[lid] / a) > majority_frac:
            excluded.add(lid)
    return excluded


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
    Reads .shape lazily (zarr/dask) without loading pixel data into RAM.
    """
    out: Dict[int, Tuple[int, int]] = {}
    for lvl in discover_pyramid_levels(multiscale):
        try:
            arr = load_level_from_multiscale(multiscale, lvl)
            # zarr / dask arrays expose .shape without loading data
            if hasattr(arr, "shape") and len(arr.shape) >= 2:
                out[int(lvl)] = (int(arr.shape[0]), int(arr.shape[1]))
            else:
                arr_np = np.asarray(arr)
                if arr_np.ndim >= 2:
                    out[int(lvl)] = (int(arr_np.shape[0]), int(arr_np.shape[1]))
        except Exception:
            continue
    return out


def lobule_miou(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    fg: np.ndarray,
    tissue: Optional[np.ndarray] = None,
    tissue_margin_px: int = 10,
    min_gt_px: int = 100,
    min_interior_px: int = 50,
    penalize_fp: bool = False,
    min_pred_px: int = 100,
) -> float:
    """Mean IoU of matched lobule instances (greedy best-match per GT lobule).

    When *tissue* is supplied, lobules touching the tissue boundary are compared
    on *interior* pixels only, those at least *tissue_margin_px* from the tissue
    edge, because assignment there is determined by where the tissue happens to
    end rather than by real lobule structure.

    IMPORTANT: this restriction is OPT-IN.  It activates only when *tissue* is
    passed; leave it out and *tissue_margin_px* and *min_interior_px* are inert,
    with boundary-touching lobules evaluated on their full extent.  Pass *tissue*
    when the ground truth covers only part of the slide, where a prediction
    clipped at the annotated edge would otherwise be scored as a bad match.

    Parameters
    ----------
    gt_labels : np.ndarray
        2-D integer GT label image (0 = background).
    pred_labels : np.ndarray
        2-D integer predicted label image (0 = background).
    fg : np.ndarray
        Boolean foreground mask (tissue, excluding vessel holes).
    tissue : np.ndarray or None
        Boolean tissue mask. When provided, pixels within
        *tissue_margin_px* of the tissue edge are excluded from the IoU
        evaluation of boundary-touching lobules. When ``None`` the original
        (full foreground) regions are used for all lobules.
    tissue_margin_px : int
        Minimum distance from the tissue boundary for a pixel to count as
        "interior". Only applied to lobules whose GT region touches the tissue
        edge.
    min_gt_px : int
        Minimum number of fg pixels a GT lobule must have to be included.
    min_interior_px : int
        Minimum number of interior pixels a boundary lobule must have after
        erosion; lobules below this threshold are skipped.
    penalize_fp : bool
        When True, predicted lobules that are never selected as the best
        match for any GT lobule are treated as false positives: each one
        contributes an IoU of 0.0 to the mean (provided it has at least
        *min_pred_px* foreground pixels).  This prevents a high mean IoU
        from being achieved by predicting only a few well-placed lobules
        while leaving large tissue areas unaccounted for.
    min_pred_px : int
        Minimum foreground pixel count for an unmatched predicted lobule to
        be counted as a false positive (only used when *penalize_fp* is True).

    Returns
    -------
    float
        Mean per-lobule IoU in [0, 1]. Returns 0.0 if no eligible GT
        lobules exist.
    """
    from scipy.ndimage import distance_transform_edt as _edt

    gt_ids = np.unique(gt_labels[fg])
    gt_ids = gt_ids[gt_ids > 0]
    if len(gt_ids) == 0 or not np.any(pred_labels[fg] > 0):
        return 0.0

    # Pre-compute interior mask once if tissue is provided.
    if tissue is not None:
        d_tissue = _edt(tissue.astype(bool)).astype(np.float32)
        interior_mask = d_tissue >= float(tissue_margin_px)
    else:
        interior_mask = None

    ious: List[float] = []
    matched_pred_ids: set = set()

    for gl in gt_ids:
        gt_r = (gt_labels == gl) & fg
        if gt_r.sum() < min_gt_px:
            continue

        # Determine evaluation region: interior-only for boundary lobules.
        if interior_mask is not None:
            gt_eval = gt_r & interior_mask
            if gt_eval.sum() < min_interior_px:
                # Lobule is almost entirely at the tissue boundary, so skip.
                continue
        else:
            gt_eval = gt_r

        # Best-matching predicted region (plurality vote inside gt_eval).
        pred_in = pred_labels[gt_eval]
        pred_in = pred_in[pred_in > 0]
        if len(pred_in) == 0:
            ious.append(0.0)
            continue
        best = int(np.bincount(pred_in).argmax())
        matched_pred_ids.add(best)

        eval_mask = interior_mask if interior_mask is not None else np.ones_like(gt_r)

        # Split-tolerant matching: if multiple pred labels each overlap >20%
        # of this GT lobule, union them for a better IoU.
        gt_px = int(gt_eval.sum())
        pred_counts_in_gt = np.bincount(pred_labels[gt_eval].ravel())
        candidate_ids = [best]
        for pid in range(1, len(pred_counts_in_gt)):
            if pid == best:
                continue
            if pred_counts_in_gt[pid] >= 0.2 * gt_px:
                candidate_ids.append(pid)
                matched_pred_ids.add(pid)

        pred_union = np.zeros_like(gt_r)
        for pid in candidate_ids:
            pred_union |= (pred_labels == pid)
        pred_union &= fg

        pred_eval = pred_union & eval_mask

        inter = int((gt_eval & pred_eval).sum())
        union = int((gt_eval | pred_eval).sum())
        ious.append(inter / max(union, 1))

    # Penalise unmatched predicted lobules (false positives).
    if penalize_fp:
        pred_ids = np.unique(pred_labels[fg])
        pred_ids = pred_ids[pred_ids > 0]
        pred_counts = np.bincount(pred_labels[fg].ravel())
        for pid in pred_ids:
            if pid not in matched_pred_ids and int(pred_counts[pid]) >= min_pred_px:
                ious.append(0.0)

    return float(np.mean(ious)) if ious else 0.0


