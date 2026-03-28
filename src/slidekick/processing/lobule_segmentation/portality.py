"""Portality mapping for lobule segmentation.

Computes a continuous portality value for each pixel inside segmented lobule
instances.  Values are in [0, 1] where:

- 0.0  — portal vessels and instance boundaries
- 1.0  — central veins
- NaN  — outside any instance

The formula is ``P = d_PB / (d_PB + d_CV)`` where *PB* is the union of
portal vessels and the instance boundary, and *d* denotes the Euclidean
distance transform.

Contours are OpenCV-style ``(x, y)`` arrays from ``cv2.findContours``.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt as edt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIVISION_GUARD: float = 1e-8
"""Small epsilon added to the denominator to avoid division by zero."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _edt_to_set(target: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest True pixel in *target*.

    Parameters
    ----------
    target : np.ndarray
        2-D boolean mask.

    Returns
    -------
    np.ndarray
        Float32 distance array.  +inf if *target* is all-False,
        zeros if all-True.
    """
    target = target.astype(bool, copy=False)
    h, w = target.shape
    if not np.any(target):
        return np.full((h, w), np.inf, dtype=np.float32)
    if np.all(target):
        return np.zeros((h, w), dtype=np.float32)
    return edt(~target).astype(np.float32)


def _rasterize_contours(
    shape: tuple[int, int],
    contours: list[np.ndarray],
) -> np.ndarray:
    """Rasterize polygon contours to a filled boolean mask.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(H, W)`` of the output mask.
    contours : list[np.ndarray]
        OpenCV-style ``(x, y)`` contours with shape ``(N, 2)``
        or ``(N, 1, 2)``.

    Returns
    -------
    np.ndarray
        Boolean mask.

    Raises
    ------
    ValueError
        If a contour does not have shape ``(N, 2)`` after squeezing.
    """
    h, w = shape
    if not contours:
        return np.zeros((h, w), dtype=bool)
    polys: list[np.ndarray] = []
    for c in contours:
        c = np.asarray(c)
        c = np.squeeze(c)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("Contour must be shaped (N, 2).")
        polys.append(c.astype(np.int32))
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, polys, 1)
    return canvas.astype(bool)


def _instance_boundary(labels: np.ndarray) -> np.ndarray:
    """8-connected boundary detection for labeled or binary masks.

    A foreground pixel is on the boundary if it touches background or a
    different label in its 8-neighborhood.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.

    Returns
    -------
    np.ndarray
        Boolean boundary mask.
    """
    lab = labels.astype(np.int32, copy=False)
    h, w = lab.shape
    pad = np.pad(lab, 1, mode="constant", constant_values=-1)
    center = pad[1: h + 1, 1: w + 1]
    boundary = np.zeros((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh = pad[1 + dy: h + 1 + dy, 1 + dx: w + 1 + dx]
            boundary |= (center > 0) & (
                (neigh != center) | (neigh == -1)
            )
    return boundary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mask_to_portality(
    mask: np.ndarray,
    cv_contours: list[np.ndarray],
    pf_contours: list[np.ndarray],
    report_path: Path | str | None = None,
) -> np.ndarray:
    """Compute the portality map for a labeled lobule mask.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer label image (0 = background, >0 = instance id).
    cv_contours : list[np.ndarray]
        Central-vein contours in OpenCV ``(x, y)`` format.
    pf_contours : list[np.ndarray]
        Portal-field (periportal vessel) contours.
    report_path : Path | str | None
        If given, writes ``portality.png`` using the magma colormap.

    Returns
    -------
    np.ndarray
        Float32 portality map with NaN outside instances.

    Raises
    ------
    ValueError
        If *mask* is not 2-D.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D (H, W).")

    h, w = mask.shape
    labels = mask.astype(np.int32, copy=False)
    inside = labels > 0

    cv_mask = _rasterize_contours((h, w), cv_contours) & inside
    pf_mask = _rasterize_contours((h, w), pf_contours) & inside
    boundary = _instance_boundary(labels) & inside
    pb_mask = (pf_mask | boundary) & inside

    d_cv = _edt_to_set(cv_mask)
    d_pb = _edt_to_set(pb_mask)

    P = (d_pb / (d_pb + d_cv + _DIVISION_GUARD)).astype(np.float32)
    P[pb_mask] = 0.0
    P[cv_mask] = 1.0
    P[~inside] = np.nan

    if report_path is not None:
        _write_portality_report(P, report_path)

    return P


def _write_portality_report(
    P: np.ndarray,
    report_path: Path | str,
) -> None:
    """Save a magma-coloured portality PNG.

    Parameters
    ----------
    P : np.ndarray
        Float32 portality map.
    report_path : Path | str
        Output directory.
    """
    outdir = Path(report_path)
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / "portality.png"
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0.0)
    arr = ma.masked_invalid(P).astype(np.float32)
    plt.imsave(
        png_path.as_posix(), arr, cmap=cmap, vmin=0.0, vmax=1.0,
    )
