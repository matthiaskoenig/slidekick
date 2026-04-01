from typing import Optional, Tuple
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes, label as ndi_label
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops


def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
    """Rearrange a 3D array to (H, W, C) layout.

    Uses the heuristic that the channel dimension is much smaller
    than the spatial dimensions (C <= 64 and C < H and C < W).
    """
    s = arr.shape
    # Channel-first: (C, H, W) where s[0] <= 64, s[1] > s[0], s[2] > s[0]
    if s[0] <= 64 and s[1] > s[0] and s[2] > s[0]:
        return np.moveaxis(arr, 0, 2)
    # Channel-last: (H, W, C) where s[2] <= 64, s[0] > s[2], s[1] > s[2]
    if s[2] <= 64 and s[0] > s[2] and s[1] > s[2]:
        return arr
    # Fallback: smallest axis is channels
    cax = int(np.argmin(s))
    if cax == 2:
        return arr
    return np.moveaxis(arr, cax, 2)


def ensure_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an input image to a uint8 grayscale image.

    Handles any combination of layout (H,W  /  H,W,C  /  C,H,W),
    dtype (uint8, uint16, float32/64), and channel count (RGB, multiplex).

    Parameters
    ----------
    image : np.ndarray
        Input image.  For 3-D inputs the channel axis is detected
        heuristically (smallest dim <= 64 that is smaller than both
        spatial dims).

    Returns
    -------
    np.ndarray
        A 2D uint8 array with values in [0, 255] suitable for thresholding.
    """
    arr = np.asarray(image)

    # --- Stage 1: ensure (H, W, C) ---
    if arr.ndim == 2:
        hwc = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        hwc = _ensure_hwc(arr)
    else:
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            hwc = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            hwc = _ensure_hwc(arr)
        else:
            raise ValueError(
                f"Cannot convert array with shape {image.shape} to grayscale."
            )

    src_dtype = hwc.dtype
    C = hwc.shape[2]

    # --- Stage 2: normalize each channel to float32 [0, 1] ---
    hwc_f = hwc.astype(np.float32)

    if np.issubdtype(src_dtype, np.integer):
        maxv = float(np.iinfo(src_dtype).max)
        if maxv > 0:
            hwc_f /= maxv
    else:
        for c in range(C):
            ch = hwc_f[:, :, c]
            p_high = float(np.nanpercentile(ch, 99))
            if p_high > 1.5:
                p_low = float(np.nanpercentile(ch, 1))
                rng = p_high - p_low
                if rng > 0:
                    hwc_f[:, :, c] = (ch - p_low) / rng
                else:
                    hwc_f[:, :, c] = 0.0

    np.clip(hwc_f, 0.0, 1.0, out=hwc_f)

    # --- Stage 3: combine channels -> single grayscale uint8 ---
    if C == 1:
        gray = hwc_f[:, :, 0]
    else:
        gray = np.mean(hwc_f, axis=2)

    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


def detect_tissue_mask(gray: np.ndarray, morphological_radius: int,
                       blur_frac: float = 0.007,
                       close_frac: float = 0.01,
                       hole_area_frac: float = 0.05,
                       min_obj_frac: float = 0.01) -> np.ndarray:
    """Compute a boolean mask of tissue regions via 2-class Otsu.

    1. Resolution-invariant Gaussian blur.
    2. 2-class Otsu threshold.
    3. Small closing to bridge vessel gaps at the tissue boundary.
    4. Border-connected BG removal (interior holes become tissue).
    5. Fill holes + remove small objects.

    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    morphological_radius : int
        Kept for backward compatibility; unused.
    blur_frac : float
        Gaussian sigma as fraction of min(H, W).
    close_frac : float
        Closing radius (for vessel bridging) as fraction of min(H, W).
    hole_area_frac : float
        Max hole area to remove, as fraction of total area.
    min_obj_frac : float
        Min object area to keep, as fraction of total area.
    """
    H, W = gray.shape[:2]
    dim = min(H, W)
    area = H * W

    # Step 1: resolution-invariant blur
    sigma = max(int(blur_frac * dim), 1)
    g_blur = cv2.GaussianBlur(gray.astype(np.uint8), (0, 0), sigma)

    # Step 2: 2-class Otsu
    try:
        thresh = threshold_otsu(g_blur)
    except Exception:
        thresh = float(np.mean(gray)) - 1.0
    m_tis = (gray > thresh)

    # Step 3: small closing to bridge vessel gaps at tissue boundary
    close_r = max(int(close_frac * dim / 2), 1)
    if close_r >= 2:
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        m_tis = cv2.morphologyEx(
            m_tis.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)

    # Step 4: border-connected BG removal
    bg_labeled, n = ndi_label(~m_tis)
    if n > 0:
        border_ids = set()
        border_ids.update(bg_labeled[0, :].ravel())
        border_ids.update(bg_labeled[-1, :].ravel())
        border_ids.update(bg_labeled[:, 0].ravel())
        border_ids.update(bg_labeled[:, -1].ravel())
        border_ids.discard(0)
        m_tis = ~np.isin(bg_labeled, list(border_ids))

    # Step 5: fill holes + remove small objects
    m_tis = binary_fill_holes(m_tis)
    hole_area = max(int(hole_area_frac * area), 1)
    m_tis = remove_small_holes(m_tis, area_threshold=hole_area)
    min_obj = max(int(min_obj_frac * area), 1)
    m_tis = remove_small_objects(m_tis, min_size=min_obj)

    return m_tis.astype(bool)


# Find the bounding box of the largest connected component in the mask
def largest_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (x, y, w, h) bounding all non-zero pixels.
    Returns None if mask contains no non-zero values.
    """
    if mask is None or not np.any(mask):
        return None

    coords = np.argwhere(mask)
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)

    x = int(minc)
    y = int(minr)
    w = int(maxc - minc + 1)
    h = int(maxr - minr + 1)

    return (x, y, w, h)


# Crop the image to the given bounding box and handle image boundaries safely
def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Return a cropped copy of `image` defined by bbox=(x,y,w,h)."""
    x, y, w, h = bbox
    y0 = max(0, int(y))
    x0 = max(0, int(x))
    y1 = min(image.shape[0], int(y + h))
    x1 = min(image.shape[1], int(x + w))
    return image[y0:y1, x0:x1].copy()
