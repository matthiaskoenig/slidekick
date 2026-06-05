"""Portality mapping for lobule segmentation.

Computes a continuous portality value for each pixel inside segmented lobule
instances. Values are in [0, 1] where:

- 0.0  - portal vessels and instance boundaries
- 1.0  - central veins
- NaN  - outside any instance

The formula is ``P = d_PB / (d_PB + d_CV)`` where *PB* is the union of
portal vessels and the instance boundary, and *d* denotes the Euclidean
distance transform.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt as edt, gaussian_filter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIVISION_GUARD: float = 1e-8
"""Small epsilon added to the denominator to avoid division by zero."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_cv_seed(
    region: np.ndarray,
    pb: np.ndarray,
    pv_stain: np.ndarray | None,
    *,
    seed_radius: int = 5,
    smooth_sigma: float = 10.0,
    min_intensity_span: float = 1e-3,
) -> np.ndarray:
    """Locate a surrogate central-vein seed when no CV annotation is present.

    Tries three strategies in order:

    1. **PV-stain intensity peak** - the smoothed maximum of *pv_stain* inside
       *region*. A central vein typically appears bright in the PV channel.
       Skipped when *pv_stain* is ``None`` or the stain is nearly flat.
    2. **Furthest-from-portal-boundary pixel** - the point inside *region* with
       the maximum Euclidean distance to *pb* (the portal boundary). This is
       the geometric centre of the lobule, equivalent to the medial axis peak.
    3. **Centroid** - a single pixel at the region centroid, used only when both
       strategies above yield an empty set (degenerate region).

    Parameters
    ----------
    region : np.ndarray
        Boolean mask of the lobule (2-D, same shape as the label image).
    pb : np.ndarray
        Boolean mask of the portal boundary for this lobule
        (lobule boundary ∪ portal vessels within the lobule).
    pv_stain : np.ndarray or None
        Float32 PV-channel image aligned with *region*.
    seed_radius : int
        Radius in pixels of the disk placed around the chosen seed point.
    smooth_sigma : float
        Gaussian smoothing sigma (px) applied to *pv_stain* in strategy 1.
    min_intensity_span : float
        Minimum smoothed peak-to-min difference required to trust the stain
        peak. Below this value the stain is treated as flat and strategy 1
        is skipped.

    Returns
    -------
    np.ndarray
        Boolean mask of seed pixels, intersected with *region*.
    """
    h, w = region.shape

    def _disk(cy: int, cx: int) -> np.ndarray:
        y0, y1 = max(0, cy - seed_radius), min(h, cy + seed_radius + 1)
        x0, x1 = max(0, cx - seed_radius), min(w, cx + seed_radius + 1)
        yg, xg = np.mgrid[y0:y1, x0:x1]
        in_disk = (yg - cy) ** 2 + (xg - cx) ** 2 <= seed_radius ** 2
        out = np.zeros((h, w), dtype=bool)
        out[yg[in_disk], xg[in_disk]] = True
        return out & region

    # Strategy 1: smoothed PV-stain intensity peak
    if pv_stain is not None:
        smoothed = gaussian_filter(
            np.where(region, pv_stain.astype(np.float32), 0.0),
            sigma=smooth_sigma,
        )
        vals = smoothed[region]
        if float(vals.max() - vals.min()) > min_intensity_span:
            masked = np.where(region, smoothed, -np.inf)
            cy, cx = np.unravel_index(int(np.argmax(masked)), (h, w))
            seed = _disk(cy, cx)
            if seed.any():
                return seed

    # Strategy 2: pixel maximally far from the portal boundary
    d_from_pb = edt(~pb).astype(np.float32)
    d_in = np.where(region, d_from_pb, -np.inf)
    cy, cx = np.unravel_index(int(np.argmax(d_in)), (h, w))
    seed = _disk(cy, cx)
    if seed.any():
        return seed

    # Strategy 3: centroid (degenerate fallback)
    ys, xs = np.where(region)
    out = np.zeros((h, w), dtype=bool)
    out[int(ys.mean()), int(xs.mean())] = True
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lobule_portality(
    labels: np.ndarray,
    cv_mask: np.ndarray,
    pf_mask: np.ndarray,
    pv_stain: np.ndarray | None = None,
) -> np.ndarray:
    """Per-lobule portality: ``p = d_pb / (d_pb + d_cv)`` per lobule.

    For each lobule, only that lobule's own central vein and boundary
    (+ portal vessels within it) contribute to the distances. This matches
    the definition in König et al. (2024) and avoids the global-EDT artefact
    where adjacent CVs pull portality values near inter-lobule boundaries.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image (0 = background, >0 = lobule id).
    cv_mask : np.ndarray
        Boolean mask of central-vein pixels.
    pf_mask : np.ndarray
        Boolean mask of portal-field pixels.
    pv_stain : np.ndarray or None
        Float32 PV-channel image aligned with *labels*, used as the first
        fallback when a lobule contains no annotated central vein. The
        smoothed intensity peak of this channel is taken as a surrogate CV
        (central veins are typically bright in the PV channel).
        When ``None`` the fallback skips straight to the geometric strategy.

    Returns
    -------
    np.ndarray
        Float32 portality map, NaN outside lobules.
        0.0 = lobule boundary / portal vessel, 1.0 = central vein.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be 2D (H, W).")
    h, w = labels.shape
    portality = np.full((h, w), np.nan, dtype=np.float32)
    cv_mask = cv_mask.astype(bool)
    pf_mask = pf_mask.astype(bool)

    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        region = labels == label_id
        if region.sum() < 10:
            continue

        # 4-connected boundary: pixels inside this lobule adjacent to a
        # different label or to background.
        boundary = np.zeros((h, w), dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(labels, -dy, 0), -dx, 1)
            boundary |= region & (shifted != label_id)

        # Portal boundary: this lobule's edge + portal vessels within it.
        pb = boundary | (pf_mask & region)

        # Central vein within this lobule; use structured fallback if none.
        cv_in = cv_mask & region
        if not cv_in.any():
            cv_in = _find_cv_seed(region, pb, pv_stain)

        d_pb = edt(~pb).astype(np.float32)
        d_cv = edt(~cv_in).astype(np.float32)

        P = (d_pb / (d_pb + d_cv + _DIVISION_GUARD)).astype(np.float32)
        portality[region] = P[region]

    # Hard-clamp vessel pixels for unambiguous downstream use.
    portality[cv_mask & (labels > 0)] = 1.0
    portality[pf_mask & (labels > 0)] = 0.0
    return portality
