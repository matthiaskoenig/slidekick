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
        Input image. Can be 2D (H, W), or 3D in either H,W,C or C,H,W layout,
        with any number of channels (RGB, RGBA, multiplex, etc.). The channel
        axis is identified as the smallest dimension. Dtype may be float (any
        range) or any integer type.

    Returns
    -------
    np.ndarray
        A 2D uint8 array with values in 0..255 suitable for Otsu thresholding.

    Notes
    -----
    Channels are collapsed via ``np.max`` (max-intensity projection) so that
    signal present in *any* channel is preserved. For float images, values
    already in [0, 1] are scaled directly; otherwise a min-max normalisation
    is applied. Non-uint8 integer images are always min-max normalised.
    """
    if image.ndim == 3:
        # Detect channel order: C is much smaller than H and W.
        # argmin == 0  →  C,H,W;  argmin == 2  →  H,W,C
        if np.argmin(image.shape) == 0:
            image = np.moveaxis(image, 0, -1)  # C,H,W → H,W,C

        # Max-intensity projection across channels → 2D
        arr = np.max(image, axis=2)
    else:
        arr = image

    # Normalise to uint8 using a unified path
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            # Standard [0, 1] float
            gray = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            # Raw float intensities — min-max normalise
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)
            if arr_max > arr_min:
                gray = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
            else:
                gray = np.zeros(arr.shape, dtype=np.uint8)
    elif arr.dtype == np.uint8:
        gray = arr.copy()
    else:
        # Integer types (uint16, int32, …) — min-max normalise
        arr_f = arr.astype(np.float32)
        arr_min = np.nanmin(arr_f)
        arr_max = np.nanmax(arr_f)
        if arr_max > arr_min:
            gray = ((arr_f - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros(arr_f.shape, dtype=np.uint8)

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
