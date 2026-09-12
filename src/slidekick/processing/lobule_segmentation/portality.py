"""Portality mapping for lobule segmentation.

Computes a continuous portality value for each pixel inside segmented lobule
instances. Values are in [0, 1] where:

- 0.0  instance boundaries and the tissue ring around portal vessels
- 1.0  the tissue ring around central veins
- NaN  outside any instance, vessel lumens included

The axis runs surface to surface, not lumen to lumen: a vessel lumen contains no
tissue, so it has no zonation value and stays NaN. The endpoints sit on the
immediate ring of tissue surrounding each detected vessel.

The formula is ``P = d_PB / (d_PB + d_CV)`` where *PB* is the union of the
instance boundary and the portal-vessel ring, *CV* is the central-vein ring, and
*d* denotes the Euclidean distance transform.
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt as edt, gaussian_filter
from scipy.ndimage import label as _ndlabel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIVISION_GUARD: float = 1e-8
"""Small epsilon added to the denominator to avoid division by zero."""

_VESSEL_ANCHOR_RING: int = 3
"""Pixels a lobule is dilated by when looking for its own vessels.

Vessel lumens are holes in the watershed foreground and therefore carry label 0,
so an un-dilated ``cv_mask & region`` never finds the lobule's central vein.
"""

_VESSEL_EDGE_RING: int = 1
"""Thickness of the tissue ring around a vessel that carries the axis endpoint.

Distinct from :data:`_VESSEL_ANCHOR_RING`, which is a *lookup* radius. This one
is the anchor itself: the immediate ring of tissue around a central vein is the
1.0 end, around a portal vessel the 0.0 end, while the lumen stays unvalued.
"""


LUMEN_DISPLAY_RGB: tuple[int, int, int] = (158, 166, 179)
"""Neutral grey used to draw vessel lumens in portality figures.

Shared so every renderer agrees. Painting lumens the same colour as the
background is actively misleading: magma's top colour is a pale cream
(``#FCFDBF``), so an unvalued lumen and a genuine portality of 1.0 become
near-indistinguishable and a central vein reads as though it were *valued* 1
rather than being a hole *ringed* by 1.
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def vessel_lumen_mask(portality: np.ndarray) -> np.ndarray:
    """NaN regions fully enclosed by valued tissue, i.e. vessel lumens.

    :func:`lobule_portality` leaves NaN in two places that mean different things:
    outside the tissue/instances, and inside vessel lumens. Renderers need to tell
    them apart: background should read as empty, a lumen as vessel with no tissue
    to zonate. A lumen is distinguished geometrically: its NaN component does not
    touch the image border.

    Parameters
    ----------
    portality : np.ndarray
        Float portality map as returned by :func:`lobule_portality`.

    Returns
    -------
    np.ndarray
        Boolean mask, True on lumen pixels.
    """
    finite = np.isfinite(portality)
    nan_lab, _ = _ndlabel(~finite)
    if nan_lab.max() == 0:
        return np.zeros_like(finite)
    edge_ids = np.unique(np.concatenate([
        nan_lab[0, :], nan_lab[-1, :], nan_lab[:, 0], nan_lab[:, -1]]))
    return ~finite & ~np.isin(nan_lab, edge_ids)


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

    1. **PV-stain intensity peak**: the smoothed maximum of *pv_stain* inside
       *region*. A central vein typically appears bright in the PV channel.
       Skipped when *pv_stain* is ``None`` or the stain is nearly flat.
    2. **Distance-weighted centroid**: the centroid of *region* weighted by each
       pixel's Euclidean distance to *pb* (the portal boundary), i.e.
       sum(coord * d) / sum(d). A plain max-distance rule picks a point on one arm
       of C- or U-shaped lobules rather than near their centre; the weighted
       centroid pulls the seed toward the bulk of the lobule interior.
    3. **Centroid**: a single pixel at the region centroid, used only when both
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

    # Strategy 2: distance-weighted centroid.
    # Using the plain furthest-from-boundary point fails for C/U-shaped lobules
    # (common at tissue edges) where the medial-axis peak lies in the concavity
    # tip, not where the CV would biologically sit. Instead, weight each pixel by
    # its distance-from-boundary so the centroid is pulled toward the fat part of
    # the lobule, robust to concavities.
    d_from_pb = edt(~pb).astype(np.float32)
    d_in = np.where(region, d_from_pb, 0.0)
    total_w = float(d_in.sum())
    if total_w > 0:
        ys, xs = np.where(region)
        cy = int(np.round(np.sum(ys * d_in[ys, xs]) / total_w))
        cx = int(np.round(np.sum(xs * d_in[ys, xs]) / total_w))
        cy = np.clip(cy, 0, h - 1)
        cx = np.clip(cx, 0, w - 1)
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
        Float32 portality map, NaN outside lobules and inside vessel lumens.
        0.0 = lobule boundary / portal-vessel ring, 1.0 = central-vein ring.

        Leaving lumens NaN keeps them out of downstream selections such as
        ``LobuleStatistics``, which picks pixels with ``isfinite``; lumens carry
        no stain, so including them would drag the outermost portality bins down.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be 2D (H, W).")
    h, w = labels.shape
    portality = np.full((h, w), np.nan, dtype=np.float32)
    cv_mask = cv_mask.astype(bool)
    pf_mask = pf_mask.astype(bool)

    # Connected components of the central-vein mask, so a lobule can adopt an
    # entire vein rather than the sliver that happens to fall inside its
    # dilation ring (see the anchor block below).
    cv_labels, _ = _ndlabel(cv_mask)

    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        region = labels == label_id
        if region.sum() < 10:
            continue

        # This lobule's own central vein, taken as whole components. A vein lies
        # in a watershed hole, so only a rim of it falls inside `grown`; adopting
        # just that rim would leave an unfilled core whose inner lip becomes a
        # spurious boundary a few px from the tissue, capping portality at ~0.76
        # (measured). Taking the whole component lifts it to ~0.99.
        grown = binary_dilation(region, iterations=_VESSEL_ANCHOR_RING)
        cv_comp = cv_mask & grown
        if cv_comp.any():
            _ids = np.unique(cv_labels[cv_comp])
            cv_comp = np.isin(cv_labels, _ids[_ids > 0])

        # The axis is anchored on the vein's outer surface, not its lumen: a
        # lumen contains no tissue and so has no zonation value to report. The
        # immediate ring of tissue around the vein is the 1.0 end.
        #
        # `cv_comp` and `cv_ring` must stay separate. The boundary trace below
        # needs the component, so the lumen counts as interior; the distance
        # transform needs the ring. Collapsing the two lets the lumen rim back
        # into `pb` while it is also the anchor, which pins P to exactly 0.5.
        cv_ring = binary_dilation(cv_comp, iterations=_VESSEL_EDGE_RING) & ~cv_mask

        # 4-connected boundary: pixels inside this lobule adjacent to a
        # different label or to background.  Explicit slicing avoids np.roll
        # wrap-around which would create false boundaries at image edges.
        #
        # The lobule's own central vein is interior, not an edge. Lumens carry
        # label 0, so without this the vein's rim lands in `pb` while also being
        # the anchor; then d_pb == d_cv near the vein and P = d_pb/(d_pb + d_cv)
        # pins to exactly 0.5, so the zonation axis spans only half its range
        # wherever a real vein is found. Treating the whole vein component as
        # interior keeps the axis full-range.
        traced = region | cv_comp
        boundary = np.zeros((h, w), dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.zeros_like(labels)
            shifted_cv = np.zeros((h, w), dtype=bool)
            src_r = slice(max(0, dy), h + min(0, dy))
            dst_r = slice(max(0, -dy), h + min(0, -dy))
            src_c = slice(max(0, dx), w + min(0, dx))
            dst_c = slice(max(0, -dx), w + min(0, -dx))
            shifted[dst_r, dst_c] = labels[src_r, src_c]
            shifted_cv[dst_r, dst_c] = cv_comp[src_r, src_c]
            boundary |= traced & (shifted != label_id) & ~shifted_cv

        # Portal boundary: this lobule's edge plus the ring around portal vessels
        # within it. Portal vessels mark the 0.0 end of the axis, so, unlike the
        # central vein, their surface belongs here; their lumen, like the CV's,
        # holds no tissue and is left unvalued.
        pf_ring = binary_dilation(pf_mask & grown,
                                  iterations=_VESSEL_EDGE_RING) & ~pf_mask
        pb = boundary | pf_ring

        # Fall back to the structured surrogate only when no vein was found. The
        # surrogate already returns a disk of tissue pixels, so it needs no ring.
        cv_anchor = cv_ring if cv_ring.any() else _find_cv_seed(region, pb, pv_stain)

        d_pb = edt(~pb).astype(np.float32)
        d_cv = edt(~cv_anchor).astype(np.float32)

        P = (d_pb / (d_pb + d_cv + _DIVISION_GUARD)).astype(np.float32)
        portality[region] = P[region]

        # Exact endpoints. The formula already yields these (d_cv == 0 on the CV
        # ring, d_pb == 0 on pb), but _DIVISION_GUARD leaves them a hair short.
        # `pb` is applied second so it wins where a vein ring touches the lobule
        # boundary; portal/boundary is the 0.0 end by convention.
        portality[cv_anchor & region] = 1.0
        portality[pb & region] = 0.0

    # Lumens are deliberately left NaN. They are holes in the watershed
    # foreground carrying label 0, so `portality[region]` never touches them, and
    # a lumen holds no tissue whose zonation could be reported. The endpoints of
    # the axis live on the vessel surfaces, assigned per lobule above.
    return portality
