"""Portality mapping for lobule segmentation.

Computes a continuous portality value for each pixel inside segmented lobule
instances. Values are in [0, 1] where:

- 0.0  instance boundaries and the tissue ring around detected portal vessels
- 1.0  the tissue ring around central veins
- NaN  outside any instance, and detected central-vein / portal lumens

The axis runs surface to surface, not lumen to lumen: a vessel lumen contains no
tissue, so it has no zonation value and stays NaN. The endpoints sit on the
immediate ring of tissue surrounding each detected vessel. Only detected central
veins and portal vessels are landmarks; a hole the detector found but never
classified is absorbed into its surrounding lobule and interpolated by position
like ordinary tissue, rather than forcing a portal-style 0.0 around itself.

The formula is ``P = d_PB / (d_PB + d_CV)`` where *PB* is the union of the
instance boundary and the portal-vessel ring, *CV* is the central-vein ring, and
*d* denotes the Euclidean distance transform.
"""

import cv2
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


def _component_centroids(mask: np.ndarray) -> np.ndarray:
    """(x, y) centroid of each connected component of a boolean mask."""
    m = mask.astype(np.uint8)
    n, _lab, _stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.empty((0, 2), dtype=np.float64)
    return cents[1:].astype(np.float64)          # drop background, (x, y)


def _vessel_network_lines(cv_mask: np.ndarray, pf_mask: np.ndarray) -> np.ndarray:
    """Rasterised non-crossing portal-central network (bool line mask).

    Reduce the central veins and portal vessels to centroids, Delaunay-triangulate
    all of them, and keep only the portal-central edges. Delaunay is planar, so the
    bipartite subset never crosses, leaving each portal linked to the central veins
    around it. Used for display alongside the portality field.
    """
    from scipy.spatial import Delaunay

    h, w = cv_mask.shape
    lines = np.zeros((h, w), dtype=bool)
    cv_pts = _component_centroids(cv_mask.astype(bool))
    pf_pts = _component_centroids(pf_mask.astype(bool))
    if len(cv_pts) == 0 or len(pf_pts) == 0:
        return lines

    pts = np.vstack([cv_pts, pf_pts])
    is_portal = np.concatenate([np.zeros(len(cv_pts), bool),
                                np.ones(len(pf_pts), bool)])

    edges: set[tuple[int, int]] = set()
    if len(pts) >= 3:
        try:
            tri = Delaunay(pts)
            for simplex in tri.simplices:
                for a, b in ((simplex[0], simplex[1]),
                             (simplex[1], simplex[2]),
                             (simplex[2], simplex[0])):
                    if is_portal[a] != is_portal[b]:      # keep portal-central only
                        edges.add((a, b) if a < b else (b, a))
        except Exception:
            # Any triangulation failure (e.g. a collinear set) falls through to
            # the nearest-central fallback below.
            edges = set()

    if not edges:
        # Degenerate input (two points, or a collinear set Delaunay rejected):
        # link each portal to its nearest central vein.
        for j in range(len(cv_pts), len(pts)):
            d = np.hypot(pts[:len(cv_pts), 0] - pts[j, 0],
                         pts[:len(cv_pts), 1] - pts[j, 1])
            i = int(np.argmin(d))
            edges.add((i, j) if i < j else (j, i))

    canvas = np.zeros((h, w), dtype=np.uint8)
    for i, j in edges:
        cv2.line(canvas,
                 (int(round(pts[i, 0])), int(round(pts[i, 1]))),
                 (int(round(pts[j, 0])), int(round(pts[j, 1]))), 1, 1)
    return canvas.astype(bool)


def _median_lobule_diameter(labels: np.ndarray) -> float:
    """Equivalent diameter of the median-area lobule, for scale-free thresholds."""
    if labels.max() <= 0:
        return 0.0
    areas = np.bincount(labels.ravel())
    areas = areas[1:]                 # drop background
    areas = areas[areas > 0]
    if areas.size == 0:
        return 0.0
    return float(2.0 * np.sqrt(np.median(areas) / np.pi))


def _label_triple_junctions(labels: np.ndarray, merge_dist: float) -> np.ndarray:
    """Points where three or more lobules meet: the portal-triad sites.

    A 3x3 patch qualifies when it contains at least three distinct non-zero
    labels. Section-edge corners (two labels plus background) are deliberately
    not included: unlike the synthetic generator, a real section edge is not a
    portal site. Nearby hits are clustered so each corner yields one point.
    Returns an (N, 2) array of (x, y).
    """
    h, w = labels.shape
    scale = 4                          # downsample; the corner count is unchanged
    small = cv2.resize(labels.astype(np.float32),
                       (max(1, w // scale), max(1, h // scale)),
                       interpolation=cv2.INTER_NEAREST).astype(np.int32)
    sh, sw = small.shape
    hits: list = []
    for y in range(1, sh - 1):
        for x in range(1, sw - 1):
            uniq = set(small[y - 1:y + 2, x - 1:x + 2].ravel().tolist())
            uniq.discard(0)
            if len(uniq) >= 3:
                hits.append((x * scale, y * scale))
    if not hits:
        return np.empty((0, 2), np.float64)
    pts = np.array(hits, np.float64)
    merged: list = []
    used = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        cluster = [pts[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and np.linalg.norm(pts[i] - pts[j]) < merge_dist:
                cluster.append(pts[j])
                used[j] = True
        merged.append(np.mean(cluster, axis=0))
    return np.array(merged, np.float64)


def _uncovered_junction_anchors(
    labels: np.ndarray,
    pf_b: np.ndarray,
    tissue_mask: np.ndarray | None,
    cap_px: float,
    corridor_px: int,
    merge_dist: float,
) -> np.ndarray:
    """Triple junctions not already served by a portal vessel in the same valley.

    "Same valley" is geodesic: a junction is covered when a detected portal
    vessel is reachable from it along the dark inter-lobule boundary corridor
    (the ridge the watershed cut), without crossing a lobule interior, within
    *cap_px* of geodesic travel. Uncovered junctions are returned as (x, y)
    points to be added as portal anchors, so lobule corners with no detected
    tract still carry a portal landmark. Junctions covered by a real vessel are
    dropped: the vessel already anchors them.
    """
    junc = _label_triple_junctions(labels, merge_dist)
    if len(junc) == 0:
        return junc
    # No portal vessel anywhere: none of the junctions can be covered.
    if not pf_b.any():
        return junc

    from skimage.segmentation import find_boundaries
    from skimage.graph import MCP_Geometric

    corridor = binary_dilation(find_boundaries(labels, mode="thick"),
                               iterations=int(max(1, corridor_px)))
    if tissue_mask is not None:
        corridor &= tissue_mask.astype(bool)

    # Portal sources, snapped onto the corridor so the geodesic starts on it.
    pf_src = pf_b & corridor
    if not pf_src.any():
        pf_src = binary_dilation(pf_b, iterations=int(max(1, corridor_px))) & corridor
    if not pf_src.any():
        return junc                    # portals sit off every valley: leave anchors

    costs = np.where(corridor, 1.0, np.inf).astype(np.float64)
    mcp = MCP_Geometric(costs)
    cum, _ = mcp.find_costs([tuple(s) for s in np.argwhere(pf_src)])

    # The junction sits on a label boundary, hence inside the corridor already;
    # this small window only absorbs the few-px rounding from junction
    # clustering. Keeping it small stops it from discounting the geodesic budget.
    h, w = labels.shape
    win = 3
    uncovered: list = []
    for x, y in junc:
        xi, yi = int(round(x)), int(round(y))
        r0, r1 = max(0, yi - win), min(h, yi + win + 1)
        c0, c1 = max(0, xi - win), min(w, xi + win + 1)
        sub = cum[r0:r1, c0:c1]
        d = float(sub.min()) if sub.size else np.inf
        if not np.isfinite(d) or d > cap_px:
            uncovered.append((x, y))
    return np.array(uncovered, np.float64) if uncovered else np.empty((0, 2), np.float64)


def vessel_network_portality(
    cv_mask: np.ndarray,
    pf_mask: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    junction_geodesic_cap_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary-free portality from vessel distances, plus the network lines.

    A second portality plane that ignores the watershed lobule boundaries and is
    driven by the detected vessels alone. Each pixel is valued by the zonation
    index used in the Xenium spatial workflow::

        Z = d_central / (d_central + d_portal)

    where *d_central* and *d_portal* are the Euclidean distances to the nearest
    central vein and nearest portal vessel. So Z = 0 on a central vein, Z = 1 on a
    portal vessel, interpolated by position in between; along a central-portal axis
    it runs 0 -> 1. The non-crossing portal-central network (Delaunay bipartite) is
    returned alongside for display and reference.

    Note the orientation is the REVERSE of the boundary-based
    :func:`lobule_portality` (which puts 1 at the central vein): this plane follows
    the Xenium zonation-index convention (0 = central/CV-associated,
    1 = portal-associated) so it is directly comparable to that analysis.

    Parameters
    ----------
    cv_mask, pf_mask : np.ndarray
        Boolean masks of detected central-vein and portal-vessel pixels.
    tissue_mask : np.ndarray or None
        When given, the index is NaN outside it; the ratio itself is defined for
        every pixel.
    labels : np.ndarray or None
        Predicted lobule label map. When given, triple junctions (where three or
        more lobules meet: the anatomical portal-triad sites) are added as portal
        anchors wherever no detected portal vessel is reachable from the junction
        along the dark inter-lobule boundary in the same valley (geodesic). This
        fills portal gaps where a tract was not detected, using the tessellation
        geometry, and never crosses a lobule interior to borrow a vessel from a
        different valley. Junctions already served by a real vessel are left to
        that vessel.
    junction_geodesic_cap_px : float or None
        Maximum geodesic distance (px, along the boundary corridor) at which a
        detected portal vessel counts as serving a junction. None auto-derives it
        from the median lobule diameter.

    Returns
    -------
    (portality, network_lines)
        *portality*: float32 in [0, 1] (0 = central vein, 1 = portal), NaN outside
        *tissue_mask* and all-NaN when no central vein or no portal anchor is
        present. *network_lines*: bool mask of the connecting lines, for overlay.
    """
    h, w = cv_mask.shape
    cv_b = cv_mask.astype(bool)
    pf_b = pf_mask.astype(bool)

    # Valley-aware junction anchoring: enrich the portal set with lobule-corner
    # triads that no detected vessel reaches along their own valley.
    if labels is not None and np.any(labels):
        _diam = _median_lobule_diameter(labels)
        _cap = (float(junction_geodesic_cap_px)
                if junction_geodesic_cap_px is not None
                else max(20.0, _diam))
        _corr = max(4, int(round(0.05 * _diam))) if _diam > 0 else 6
        _merge = max(10.0, 0.15 * _diam) if _diam > 0 else 40.0
        try:
            _anchors = _uncovered_junction_anchors(
                labels, pf_b, tissue_mask, _cap, _corr, _merge)
        except Exception:
            _anchors = np.empty((0, 2), np.float64)
        if len(_anchors):
            _pf_u8 = pf_b.astype(np.uint8)
            _r = max(2, int(round(0.02 * _diam))) if _diam > 0 else 4
            for _x, _y in _anchors:
                cv2.circle(_pf_u8, (int(round(_x)), int(round(_y))), _r, 1, -1)
            pf_b = _pf_u8.astype(bool)

    lines = _vessel_network_lines(cv_b, pf_b)

    if not cv_b.any() or not pf_b.any():
        return np.full((h, w), np.nan, dtype=np.float32), lines

    d_cv = edt(~cv_b).astype(np.float32)
    d_pv = edt(~pf_b).astype(np.float32)
    port = (d_cv / (d_cv + d_pv + _DIVISION_GUARD)).astype(np.float32)
    # Exact endpoints (the ratio already yields these; assign for clarity).
    port[cv_b] = 0.0
    port[pf_b] = 1.0
    if tissue_mask is not None:
        port[~tissue_mask.astype(bool)] = np.nan
    return port, lines


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
        Float32 portality map, NaN outside lobules and inside detected CV/portal
        lumens. 0.0 = lobule boundary / detected-portal-vessel ring, 1.0 =
        central-vein ring. Holes the detector left unclassified are relabelled into
        the surrounding lobule and receive an interpolated value, so only the two
        detected vessel types are landmarks.

        Leaving detected lumens NaN keeps them out of downstream selections such as
        ``LobuleStatistics``, which picks pixels with ``isfinite``; lumens carry
        no stain, so including them would drag the outermost portality bins down.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be 2D (H, W).")
    h, w = labels.shape
    portality = np.full((h, w), np.nan, dtype=np.float32)
    cv_mask = cv_mask.astype(bool)
    pf_mask = pf_mask.astype(bool)

    # Absorb unclassified holes into the surrounding lobule.
    #
    # The watershed carries every detected hole as label 0 (noise specks and
    # missed vessels as well as the classified CV and portal lumens). Only
    # detected central veins and portal vessels are real landmarks: a CV ring is
    # the 1.0 end of the axis, a portal ring the 0.0 end, and both lumens are left
    # NaN. An unclassified hole is none of those and must not act as a portal, so
    # it is relabelled into its nearest lobule; it then receives an interpolated
    # value by position like ordinary tissue and no longer forces a 0.0 boundary
    # around itself. Enclosed holes are found by topology (label-0 regions that do
    # not reach an image border, so the true exterior background is left alone).
    _bg = labels == 0
    if _bg.any():
        _bg_lbl, _n_bg = _ndlabel(_bg)
        if _n_bg:
            _border = np.unique(np.concatenate([
                _bg_lbl[0, :], _bg_lbl[-1, :], _bg_lbl[:, 0], _bg_lbl[:, -1]]))
            _border = _border[_border > 0]
            _interior_holes = _bg & ~np.isin(_bg_lbl, _border)
            _unclassified = _interior_holes & ~cv_mask & ~pf_mask
            _tissue = labels > 0
            if _unclassified.any() and _tissue.any():
                _, (_iy, _ix) = edt(~_tissue, return_indices=True)
                labels = labels.copy()
                labels[_unclassified] = labels[_iy[_unclassified], _ix[_unclassified]]

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

    # Detected CV and portal lumens are deliberately left NaN. They are holes in
    # the watershed foreground carrying label 0, so `portality[region]` never
    # touches them, and a lumen holds no tissue whose zonation could be reported.
    # The endpoints of the axis live on the vessel surfaces, assigned per lobule
    # above. Unclassified holes were relabelled into a lobule at the top, so they
    # are valued here rather than left NaN.
    return portality
