"""
Portality mapping.

Outside any instance is NaN.
Central veins are 1.0.
Portal vessels and instance boundaries are 0.0.
Interior values are in [0, 1] using P = d_PB / (d_PB + d_CV) where PB is the union of portal vessels and the instance boundary.
Contours are OpenCV-style (x, y) from cv2.findContours.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import cv2
import numpy.ma as ma
import matplotlib.pyplot as plt


def _edt_to_set(target: np.ndarray) -> np.ndarray:
    """
    Euclidean distance to the nearest True in target computed by EDT on the inverted set.
    Returns +inf if target has no True. Returns zeros if target is all True.
    """
    target = target.astype(bool, copy=False)
    h, w = target.shape
    if not np.any(target):
        return np.full((h, w), np.inf, dtype=np.float32)
    if np.all(target):
        return np.zeros((h, w), dtype=np.float32)
    return edt(~target).astype(np.float32)


def _rasterize_contours(shape: Tuple[int, int], contours: List[np.ndarray]) -> np.ndarray:
    """
    Rasterize polygon contours to a filled boolean mask using OpenCV fillPoly.
    Input vertices are (x, y) with shape (N, 2) or (N, 1, 2).
    """
    h, w = shape
    if not contours:
        return np.zeros((h, w), dtype=bool)
    polys: List[np.ndarray] = []
    for c in contours:
        c = np.asarray(c)
        c = np.squeeze(c)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("Contour must be shaped (N, 2).")
        xy = c.astype(np.int32)
        polys.append(xy)
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, polys, 1)
    return canvas.astype(bool)


def _instance_boundary(labels: np.ndarray) -> np.ndarray:
    """
    8-connected boundary for labeled or binary masks.
    True where a foreground pixel touches background or a different label.
    """
    lab = labels.astype(np.int32, copy=False)
    h, w = lab.shape
    pad = np.pad(lab, 1, mode="constant", constant_values=-1)
    center = pad[1 : h + 1, 1 : w + 1]
    boundary = np.zeros((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh = pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            boundary |= (center > 0) & ((neigh != center) | (neigh == -1))
    return boundary


def mask_to_portality(
    mask: np.ndarray,
    cv_contours: List[np.ndarray],
    pf_contours: List[np.ndarray],
    report_path: Optional[Path | str] = None,
) -> np.ndarray:
    """
    Compute the portality map with portal vessels and boundaries at 0 and central veins at 1.
    If report_path is given, writes report_path/portality.png using magma in [0, 1] with NaNs transparent.
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

    eps = 1e-8
    P = (d_pb / (d_pb + d_cv + eps)).astype(np.float32)

    P[pb_mask] = 0.0
    P[cv_mask] = 1.0
    P[~inside] = np.nan

    if report_path is not None:
        outdir = Path(report_path)
        outdir.mkdir(parents=True, exist_ok=True)
        png_path = outdir / "portality.png"
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(alpha=0.0)
        arr = ma.masked_invalid(P).astype(np.float32)
        plt.imsave(png_path.as_posix(), arr, cmap=cmap, vmin=0.0, vmax=1.0)

    return P
