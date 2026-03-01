from typing import Optional, Tuple
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk
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


# Compute a binary tissue mask using Otsu thresholding and morphological closing
def detect_tissue_mask(gray: np.ndarray, morphological_radius: int) -> np.ndarray:
    """Compute a boolean mask of tissue regions from a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    morphological_radius : int
        Radius of the structuring element used for morphological closing.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates tissue.
    """
    # Use Otsu thresholding; if Otsu fails (constant image), fall back to mean
    try:
        thresh = threshold_otsu(gray)
    except Exception:
        thresh = float(np.mean(gray)) - 1.0

    binary = gray > thresh
    selem = disk(int(morphological_radius))
    closed = closing(binary, selem)
    return closed.astype(bool)


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
