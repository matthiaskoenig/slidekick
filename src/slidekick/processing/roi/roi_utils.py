from typing import Optional, Tuple
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk
from skimage.measure import label, regionprops


# Convert image to uint8 grayscale suitable for Otsu thresholding
def ensure_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an input image to a uint8 grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Input image. Can be single-channel, multi-channel (RGB/RGBA), or
        floating-point in range 0..1.

    Returns
    -------
    np.ndarray
        A 2D uint8 array with values in 0..255 suitable for thresholding.
    """
    if image.ndim == 3:
        arr = image
        # Scale float images to 0..255 and convert types
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip((arr * 255.0).astype(np.float32), 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # If at least three channels exist use RGB->grayscale luminance, else average
        if arr.shape[2] >= 3:
            r = arr[..., 0].astype(np.float32)
            g = arr[..., 1].astype(np.float32)
            b = arr[..., 2].astype(np.float32)
            gray = (0.2126 * r + 0.7152 * g + 0.0722 * b)
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        else:
            gray = np.mean(arr, axis=2).astype(np.uint8)
    else:
        # Single-channel input handling
        if np.issubdtype(image.dtype, np.floating):
            gray = np.clip((image * 255.0).astype(np.float32), 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            arr = image.astype(np.float32)
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)
            if arr_max > arr_min:
                gray = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
            else:
                gray = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            gray = image.copy()
    return gray


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
    """Return (x, y, w, h) of the largest connected component or None if none present."""
    if mask is None or np.count_nonzero(mask) == 0:
        return None

    lbl = label(mask)
    props = regionprops(lbl)
    if not props:
        return None

    largest = max(props, key=lambda p: p.area)
    minr, minc, maxr, maxc = largest.bbox
    x = int(minc)
    y = int(minr)
    w = int(maxc - minc)
    h = int(maxr - minr)
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
