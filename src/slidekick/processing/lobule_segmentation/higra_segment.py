"""Higra seeded watershed segmentation of liver lobules.

Public API
----------
segment_lobules(tissue, vessel_holes, pv_stain, *, ...)
    Main entry point. Returns an integer label map (np.int32) of the same
    spatial extent as *tissue*.

All other names in this module are private helpers (underscore prefix).
"""

from __future__ import annotations

import logging
import numpy as np
import cv2
from collections import defaultdict
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage import distance_transform_edt as _edt
from scipy.interpolate import splprep, splev
from scipy.ndimage import label as _ndlabel
from scipy.ndimage import binary_dilation, maximum_filter
from scipy.ndimage import binary_fill_holes as _fill_holes
from skimage.filters import threshold_otsu


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _seed_centroids(seed_cnts: list) -> np.ndarray:
    """Centroid (x, y) of each OpenCV seed contour.

    Returns an (N, 2) float64 array; empty (0, 2) when there are no seeds.
    """
    pts = []
    for cnt in seed_cnts or []:
        c = np.asarray(cnt).reshape(-1, 2)
        if c.size:
            pts.append(c.mean(axis=0))
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _masked_smooth(
    pv_u8_or_float: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Masked Gaussian: smooth(I * mask) / smooth(mask).

    Avoids edge artifacts that arise when tissue borders abruptly drop to zero.
    Returns a float32 array normalised to [0, 1] inside *mask*; pixels outside
    *mask* are set to zero.

    Parameters
    ----------
    pv_u8_or_float:
        Input intensity image (any numeric dtype).
    mask:
        Boolean foreground mask.
    sigma:
        Gaussian standard deviation in pixels.
    """
    pv_f = pv_u8_or_float.astype(np.float32).copy()
    pv_f[~mask] = 0.0
    mask_f = mask.astype(np.float32)
    numerator = gaussian_filter(pv_f, sigma=sigma)
    denominator = np.maximum(gaussian_filter(mask_f, sigma=sigma), 1e-6)
    smoothed = numerator / denominator
    smoothed[~mask] = 0.0
    mx = float(np.percentile(smoothed[mask], 99.5)) if mask.any() else 1.0
    if mx > 0:
        smoothed = smoothed / mx
        np.clip(smoothed, 0.0, 1.0, out=smoothed)
    return smoothed


def _make_markers_from_cnts(
    seed_cnts: list,
    H: int,
    W: int,
) -> np.ndarray:
    """Build a marker image (int32) from a list of contours.

    Each contour is filled with a unique label 1..N. Pixels not covered by
    any contour are labelled 0 (unseeded).

    Parameters
    ----------
    seed_cnts:
        List of contours in OpenCV format (N, 1, 2) int32 arrays.
    H, W:
        Image dimensions.
    """
    markers = np.zeros((H, W), np.int32)
    for i, cnt in enumerate(seed_cnts, 1):
        cv2.drawContours(markers, [cnt], -1, i, cv2.FILLED)
    return markers


def _find_intensity_cluster_seeds(
    pv_stain: np.ndarray,
    fg: np.ndarray,
    existing_seed_mask: np.ndarray,
    blob_sigma: float = 30.0,
    peak_min_dist: int = 80,
    seed_radius: int | None = None,
    peak_thresh_pct: float = 30.0,
    tissue_boundary_dist: np.ndarray | None = None,
    cluster_merge_sigma: float | None = None,
    cv_dist_map: np.ndarray | None = None,
    min_seed_border_dist: int = 0,
    peak_thresh_factor: float = 0.0,
) -> list:
    """Find bright PV-stain peaks as Voronoi seeds.

    Detects local maxima in the masked-Gaussian-smoothed PV channel, applies
    non-maximum suppression, and optionally merges peaks that share the same
    coarse-scale intensity basin (preventing vessel-hole artefacts from
    splitting a single lobule into multiple seeds).

    Parameters
    ----------
    pv_stain:
        Float32 PV-channel intensity image.
    fg:
        Boolean foreground mask (tissue minus vessel holes).
    existing_seed_mask:
        Boolean mask of pixels already claimed by other seed types. Candidate
        peaks that fall on existing seeds are discarded.
    blob_sigma:
        Gaussian sigma for fine-scale peak detection. Should approximate the
        lobule radius in pixels.
    peak_min_dist:
        Minimum pixel distance between kept peaks (NMS radius), the same
        distance everywhere, including within ``peak_min_dist // 3`` of the
        tissue edge. It used to be halved there, which placed most of the seeds
        in that edge zone, not a narrow border band but roughly two-thirds of
        the tissue. Only the brightness threshold still relaxes near the edge
        (below).
    seed_radius:
        Radius of the circular seed contour painted around each peak.
        Defaults to ``max(10, peak_min_dist // 4)``.
    peak_thresh_pct:
        Percentile brightness threshold for interior peaks. Peaks within
        ``peak_min_dist // 3`` of the tissue edge use half this threshold
        (minimum 5th percentile), to recover genuinely dim edge lobules.
    tissue_boundary_dist:
        Precomputed distance-transform-from-tissue-boundary array. Computed
        internally when *None*.
    cluster_merge_sigma:
        Gaussian sigma for the coarse-scale field used to define lobule basins.
        Should be large enough to blur over vessel holes (typically 2–4 times
        *blob_sigma*). When *None* or ≤ 0, cluster merging is disabled. Fine-
        scale peaks that map to the same coarse-scale local maximum (within
        *peak_min_dist*) are merged to the single brightest fine peak.
    min_seed_border_dist:
        Minimum distance from the tissue boundary for intensity-peak seeds.
        Peaks closer than this are rejected. Prevents false-positive seeds
        on the tissue edge.

    Returns
    -------
    list of np.ndarray
        Circular seed contours (one per kept peak) in OpenCV format.
    """
    H, W = pv_stain.shape
    if seed_radius is None:
        seed_radius = max(10, peak_min_dist // 4)

    if tissue_boundary_dist is None:
        # Measured from the hole-filled foreground, not `fg` itself. `fg` is
        # `tissue & ~vessel_holes`, so every vessel lumen is a gap in it and the
        # distance transform reads ~0 at each lumen rim, which is exactly where the
        # PV peaks are, PV being a pericentral marker. Using `fg` directly
        # therefore deletes the seeds the guard most needs to keep, collapsing the
        # lobule count. Filling the holes first holds the count stable across the
        # useful range of `min_seed_border_dist`. `_seeds_from_cv_mask` already
        # measures from a filled mask for the same reason; this path was missed.
        # The defect is invisible on real slides, where lumens rarely form holes at
        # all, which is why it shows up on synthetic data.
        #
        # Note: `min_seed_border_dist` was tuned against the un-filled mask, so its
        # shipped value is not a measured optimum under this (correct) distance.
        tissue_boundary_dist = _edt(_fill_holes(fg)).astype(np.float32)

    # Step 1: masked Gaussian smooth for fine-scale peak detection
    blurred = _masked_smooth(pv_stain, fg, blob_sigma)

    # Step 2: find local maxima at two scales
    # Exclude pixels too close to the tissue boundary.
    #
    # The edge zone used to span the full peak_min_dist and halve the NMS spacing
    # inside it, which put roughly two-thirds of the tissue in a densely packed
    # relaxed band rather than a narrow border strip. Narrowing the zone to a third
    # of peak_min_dist and keeping full spacing there (only the brightness
    # threshold still relaxes, which is what recovers dim edge lobules) leaves the
    # score unchanged while cutting the surplus edge seeds that bought no real
    # coverage.
    border_ok = tissue_boundary_dist >= min_seed_border_dist if min_seed_border_dist > 0 else fg
    edge_zone = (tissue_boundary_dist < max(3, peak_min_dist // 3)) & border_ok

    # Optional: CV-distance info for zone-aware threshold and combined NMS score.
    # cv_far = fg pixels far from any CV and not in the tissue edge zone, the
    # territory where CV-dropout lobules live and where the brightness threshold
    # must be relaxed to detect their weak PV peak.
    cv_dist_f: np.ndarray | None = None
    cv_dist_n: np.ndarray | None = None
    cv_far: np.ndarray | None = None
    if cv_dist_map is not None:
        cv_dist_f = cv_dist_map.astype(np.float32)
        cv_max = float(cv_dist_f[fg].max()) if fg.any() else 1.0
        cv_dist_n = cv_dist_f / (cv_max + 1e-6)
        cv_far = fg & ~edge_zone & (cv_dist_f > peak_min_dist)

    # NMS spacing is the same full peak_min_dist inside and outside the edge zone;
    # only the brightness threshold differs there (below). Halving the spacing near
    # the edge was the main driver of the seed over-density noted above;
    # local_max_full covers both regions, so there is only one spacing.
    local_max_full = maximum_filter(blurred, size=peak_min_dist)
    peaks_interior = (blurred == local_max_full) & (blurred > 0) & fg & ~edge_zone & border_ok
    peaks_edge = (blurred == local_max_full) & (blurred > 0) & fg & edge_zone
    peaks = peaks_interior | peaks_edge

    fg_vals = blurred[fg & (blurred > 0)]
    if len(fg_vals) == 0:
        return []

    if peak_thresh_factor > 0:
        # Adaptive threshold: peak must exceed local_mean + factor * local_std.
        # This detects dim lobule peaks that a global percentile would drop.
        _adapt_sigma = blob_sigma * 2.0
        _local_mean = _masked_smooth(blurred, fg, _adapt_sigma)
        _local_sq = _masked_smooth(blurred ** 2, fg, _adapt_sigma)
        _local_std = np.sqrt(np.maximum(_local_sq - _local_mean ** 2, 1e-6))
        _adaptive_thr = _local_mean + peak_thresh_factor * _local_std
        _adaptive_thr_edge = _local_mean + peak_thresh_factor * 0.5 * _local_std
        peaks = peaks & (
            ((blurred > _adaptive_thr) & ~edge_zone) |
            ((blurred > _adaptive_thr_edge) & edge_zone)
        )
    else:
        thr_interior = np.percentile(fg_vals, peak_thresh_pct)
        thr_edge = np.percentile(fg_vals, max(5.0, peak_thresh_pct * 0.5))

        if cv_dist_map is not None and cv_far is not None:
            far_vals = blurred[cv_far & (blurred > 0)] if cv_far.any() else np.array([], dtype=np.float32)
            thr_far = float(np.percentile(far_vals, max(5.0, peak_thresh_pct * 0.3))) if len(far_vals) else 0.0
            peaks = peaks & (
                ((blurred > thr_interior) & ~edge_zone & ~cv_far) |
                ((blurred > thr_far) & ~edge_zone & cv_far) |
                ((blurred > thr_edge) & edge_zone)
            )
        else:
            peaks = peaks & (
                ((blurred > thr_interior) & ~edge_zone) |
                ((blurred > thr_edge) & edge_zone)
            )

    # Step 3: NMS, sort by combined score and suppress nearby duplicates.
    # When cv_dist_map is available, weight brightness by distance from the
    # nearest CV so that a well-centred (far-from-CV) peak beats a slightly
    # brighter but CV-adjacent peak, for better coverage of unclaimed territory.
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return []

    if cv_dist_map is not None and cv_dist_n is not None:
        nms_score = blurred[ys, xs] * (1.0 + cv_dist_n[ys, xs])
    else:
        nms_score = blurred[ys, xs]
    order = np.argsort(-nms_score)
    ys, xs = ys[order], xs[order]

    kept_peaks: list[tuple[int, int]] = []
    for y, x in zip(ys, xs):
        if existing_seed_mask[y, x]:
            continue
        is_edge = tissue_boundary_dist[y, x] < peak_min_dist
        min_dist_sq = (peak_min_dist * 0.5) ** 2 if is_edge else float(peak_min_dist) ** 2
        too_close = any(
            (y - py) ** 2 + (x - px) ** 2 < min_dist_sq
            for py, px in kept_peaks
        )
        if not too_close:
            kept_peaks.append((y, x))

    # Step 4: coarse-basin cluster-merge
    if cluster_merge_sigma and cluster_merge_sigma > 0 and len(kept_peaks) > 1:
        coarse = _masked_smooth(pv_stain, fg, cluster_merge_sigma)

        coarse_maxfilt = maximum_filter(coarse, size=peak_min_dist)
        coarse_peak_mask = (coarse == coarse_maxfilt) & (coarse > 0) & fg
        cp_ys, cp_xs = np.where(coarse_peak_mask)

        if len(cp_ys) > 0:
            groups: dict = defaultdict(list)
            for y, x in kept_peaks:
                dists_sq = (cp_ys - y) ** 2 + (cp_xs - x) ** 2
                nearest = int(np.argmin(dists_sq))
                groups[nearest].append((blurred[y, x], y, x))

            kept_peaks = []
            for group in groups.values():
                best = max(group, key=lambda t: t[0])
                kept_peaks.append((best[1], best[2]))

    # Step 5: create circular seed contours
    new_seeds = []
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    for y, x in kept_peaks:
        pts = np.stack(
            [
                np.clip(x + seed_radius * np.cos(angles), 0, W - 1),
                np.clip(y + seed_radius * np.sin(angles), 0, H - 1),
            ],
            axis=-1,
        ).astype(np.int32).reshape(-1, 1, 2)
        new_seeds.append(pts)

    return new_seeds


def _seeds_from_cv_mask(
    cv_mask: np.ndarray,
    tissue: np.ndarray,
    seed_tissue_ring: int = 8,
    min_cv_area: int = 50,
    min_border_dist: int = 0,
) -> list:
    """One circular seed per connected CV component.

    The seed circle is centred at the component centroid and sized so that its
    outer ring always extends *seed_tissue_ring* pixels into the surrounding
    tissue, regardless of CV size.  This guarantees that some marker pixels are
    in the foreground (tissue & ~vessel_holes), giving the watershed a proper
    low-inverted-PV starting point, identical in spirit to the intensity-peak
    seeds that worked well before. Seeding entirely inside the vessel lumen
    (where inverted PV is high) caused hg.WeightFunction.max to flatten the
    cost landscape and degraded boundary quality.

    Parameters
    ----------
    cv_mask : np.ndarray
        Boolean mask of central vein lumens.
    tissue : np.ndarray
        Boolean tissue mask (used for shape reference only).
    seed_tissue_ring : int
        How many pixels beyond the CV boundary the seed circle must reach.
        seed_radius per component = estimated_CV_radius + seed_tissue_ring.
    min_cv_area : int
        Minimum component area in pixels.  Smaller components are treated as
        noise and skipped (no seed produced).
    min_border_dist : int
        Minimum distance from the tissue boundary for a CV centre to be
        seeded.  Previously absent entirely: `min_seed_border_dist` guarded
        only the intensity-peak seeds, so a vessel detected in a concavity of
        the tissue outline became a seed unconditionally and produced a marker
        sitting on the tissue edge.  0 restores the old behaviour.

    Returns
    -------
    list of np.ndarray
        Circular seed contours in OpenCV (N, 1, 2) int32 format, one per CV component.
    """
    H, W = cv_mask.shape
    labeled_cv, n_cv = _ndlabel(cv_mask)

    seed_cnts: list = []
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)

    # Distance from the OUTER tissue border, used both to reject border-hugging
    # CVs and to keep the seed disc inside the tissue.
    #
    # This must be measured on the hole-filled tissue. A CV lumen carries no
    # stain, so it is not part of `tissue` itself; _edt(tissue) is therefore 0
    # at every CV centre, and a naive `dist < min_border_dist` test would reject
    # every single CV seed.
    _tis_dist = None
    if tissue.any():
        _tis_dist = _edt(_fill_holes(tissue))

    for comp_id in range(1, n_cv + 1):
        ys = np.where(labeled_cv == comp_id)[0]
        if len(ys) < min_cv_area:               # skip noise pixels / tiny artefacts
            continue

        # Inscribed-circle centre via distance-transform maximum.
        # For a perfect circle this equals the centroid; for elongated or
        # irregular lumens it is the most interior point, geometrically the
        # true centre of the lumen, which is what the watershed needs as its
        # anchor so that portality flows outward from the correct origin.
        comp_mask = (labeled_cv == comp_id)
        _dt_cv = _edt(comp_mask)
        cy, cx = divmod(int(np.argmax(_dt_cv)), W)

        # Inscribed-circle radius (EDT value at the centre) is more accurate
        # than the area-based approximation, especially for elongated CVs.
        # Reject CVs that sit in a notch of the tissue outline rather than in
        # the tissue body.
        if min_border_dist > 0 and _tis_dist is not None:
            if float(_tis_dist[cy, cx]) < float(min_border_dist):
                continue

        comp_radius = max(1, int(np.ceil(float(_dt_cv[cy, cx]))))
        seed_radius = comp_radius + seed_tissue_ring

        # Cap the disc so it cannot extend past the tissue boundary.  It was
        # previously clipped to image bounds only, so seed_tissue_ring (20 px
        # in the tuned baseline) pushed the marker bodily outside the tissue
        # for any CV near the edge.
        if _tis_dist is not None:
            _room = int(_tis_dist[cy, cx])
            if _room > 0:
                seed_radius = min(seed_radius, max(comp_radius, _room - 1))

        pts = np.stack(
            [
                np.clip(cx + seed_radius * np.cos(angles), 0, W - 1),
                np.clip(cy + seed_radius * np.sin(angles), 0, H - 1),
            ],
            axis=-1,
        ).astype(np.int32).reshape(-1, 1, 2)
        seed_cnts.append(pts)

    return seed_cnts


def _merge_below_valley(
    labels: np.ndarray,
    smoothed_pv: np.ndarray,
    fg: np.ndarray,
    valley_thr: float,
    protected_labels: frozenset | None = None,
    pp_smooth: np.ndarray | None = None,
    pp_thr: float = 0.5,
    protected_merge_pv_thr: float | None = None,
) -> np.ndarray:
    """Merge adjacent lobule pairs that share no real PV valley on their boundary.

    Two adjacent labelled regions A and B are merged when the mean smoothed-PV
    value along their shared pixel interface is >= *valley_thr* - meaning no dark
    inter-lobule region separates them, so they are most likely one lobule that was
    over-segmented.  Using the mean (instead of the minimum) is robust against
    single dark pixels (noise, vessel fragments) that would otherwise block merging.

    **Protected pairs** (both labels contain a CV): protected labels are normally
    skipped to avoid merging correctly-seeded lobules.  Exception: when *pp_smooth*
    is provided and the maximum PP signal along the shared boundary is below
    *pp_thr*, both PV and PP agree there is no portal tract, so the pair is a fused
    super-lobule and is merged. This catches the synthetic fusion case where two
    CVs belong to one biological unit.

    Uses union-find for transitive merging (A+B and B+C -> A+B+C).

    Parameters
    ----------
    labels : np.ndarray
        Integer label map (0 = background, >0 = lobule id).
    smoothed_pv : np.ndarray
        Masked-Gaussian-smoothed PV stain (float32, [0, 1] in *fg*).
        High values = lobule interior; low values = dark valley.
    fg : np.ndarray
        Boolean foreground mask.
    valley_thr : float
        Minimum smoothed-PV below which a pixel is considered a real valley.
        Computed externally via threshold_otsu(smoothed_pv[fg]).
    protected_labels : frozenset or None
        Label IDs seeded from annotated CVs.  Protected pairs are only merged
        when *pp_smooth* confirms absence of a portal tract.
    pp_smooth : np.ndarray or None
        Fine-scale smoothed PP stain (float32, [0, 1] in *fg*).  High values
        = portal tract = lobule boundary.  When provided, used as a second
        channel to confirm or veto merges of protected pairs.
    pp_thr : float
        PP threshold above which a boundary pixel is considered a portal-tract
        pixel. Protected pairs whose max boundary PP >= *pp_thr* are not
        merged (real portal tract detected).

    Returns
    -------
    np.ndarray
        int32 label map with merged regions.
    """
    labels = labels.copy().astype(np.int32)
    H, W   = labels.shape
    max_id = int(labels.max()) + 1
    if max_id <= 1:
        return labels

    pair_min: dict[int, float] = {}
    pair_sum: dict[int, float] = {}   # running sum of seam PV (for mean)
    pair_cnt: dict[int, int]   = {}   # boundary-pixel count per pair
    pair_max_pp: dict[int, float] = {}   # max PP on boundary per pair (portal-tract evidence)

    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # Slice indices for the shift (handles negative dx for diagonal)
        if dy == 0:
            row_a, row_b = slice(None), slice(None)
        else:
            row_a, row_b = slice(None, H - dy), slice(dy, None)
        if dx > 0:
            col_a, col_b = slice(None, W - dx), slice(dx, None)
        elif dx < 0:
            col_a, col_b = slice(-dx, None), slice(None, W + dx)
        else:
            col_a, col_b = slice(None), slice(None)

        la  = labels    [row_a, col_a]
        lb  = labels    [row_b, col_b]
        pv_a = smoothed_pv[row_a, col_a]
        pv_b = smoothed_pv[row_b, col_b]

        bnd = (la != lb) & (la > 0) & (lb > 0)
        if not bnd.any():
            continue

        la_b   = la[bnd].astype(np.int64)
        lb_b   = lb[bnd].astype(np.int64)
        # Conservative: minimum PV on either side of the interface
        pv_bnd = np.minimum(pv_a[bnd], pv_b[bnd])
        # Average PV on either side, robust to a single dark pixel pulling
        # the min down (the watershed cut always finds the darkest path).
        pv_mean_bnd = 0.5 * (pv_a[bnd].astype(np.float64) + pv_b[bnd].astype(np.float64))

        # PP: maximum on either side (high = portal tract present)
        pp_bnd = None
        if pp_smooth is not None:
            pp_a = pp_smooth[row_a, col_a]
            pp_b = pp_smooth[row_b, col_b]
            pp_bnd = np.maximum(pp_a[bnd], pp_b[bnd])

        # Canonicalise pair (a <= b) -> unique int64 code
        swap          = la_b > lb_b
        la_b[swap], lb_b[swap] = lb_b[swap].copy(), la_b[swap].copy()
        codes         = la_b * max_id + lb_b

        unique_codes, inv = np.unique(codes, return_inverse=True)
        min_pvs = np.full(len(unique_codes), np.inf, dtype=np.float64)
        np.minimum.at(min_pvs, inv, pv_bnd)
        sum_pvs = np.zeros(len(unique_codes), dtype=np.float64)
        np.add.at(sum_pvs, inv, pv_mean_bnd)
        cnt_pvs = np.zeros(len(unique_codes), dtype=np.int64)
        np.add.at(cnt_pvs, inv, 1)

        max_pps = None
        if pp_bnd is not None:
            max_pps = np.full(len(unique_codes), -np.inf, dtype=np.float64)
            np.maximum.at(max_pps, inv, pp_bnd)

        for i, (code, min_pv) in enumerate(zip(unique_codes.tolist(), min_pvs.tolist())):
            code = int(code)
            if code in pair_min:
                if min_pv < pair_min[code]:
                    pair_min[code] = float(min_pv)
                pair_sum[code] += float(sum_pvs[i])
                pair_cnt[code] += int(cnt_pvs[i])
            else:
                pair_min[code] = float(min_pv)
                pair_sum[code] = float(sum_pvs[i])
                pair_cnt[code] = int(cnt_pvs[i])

            if max_pps is not None:
                max_pp = float(max_pps[i])
                if code in pair_max_pp:
                    if max_pp > pair_max_pp[code]:
                        pair_max_pp[code] = max_pp
                else:
                    pair_max_pp[code] = max_pp

    ids    = np.unique(labels[labels > 0])
    parent = {int(i): int(i) for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for code, min_pv in pair_min.items():
        a = int(code // max_id)
        b = int(code % max_id)
        mean_pv = pair_sum[code] / max(pair_cnt[code], 1)
        is_protected = bool(protected_labels and
                            (a in protected_labels or b in protected_labels))
        pp_ok = (pp_smooth is None or pair_max_pp.get(code, 1.0) < pp_thr)

        merge = False
        if not is_protected and mean_pv >= valley_thr:
            merge = True
        elif (protected_merge_pv_thr is not None and is_protected
              and pp_smooth is None and mean_pv >= protected_merge_pv_thr):
            # No-PP: merge protected pair when mean seam PV is high enough
            # (no dark periportal valley).  PP instances are excluded because
            # PP smoothing makes reliable fused-seam detection impractical.
            merge = True
        elif (is_protected and protected_merge_pv_thr is None
              and pp_smooth is not None and pp_ok and min_pv >= valley_thr):
            merge = True

        if merge and a in parent and b in parent:
            union(a, b)

    result = labels.copy()
    for old_id in ids:
        root = find(int(old_id))
        if root != int(old_id):
            result[labels == old_id] = root

    result[~fg] = 0
    return result


def _smooth_labels_valley_aware(
    labels: np.ndarray,
    inverted: np.ndarray,
    fg: np.ndarray,
    sigma: float = 5.0,
    valley_pct: float = 75.0,
    valley_band_px: int = 2,
    pp_smooth: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth label boundaries while preserving intensity valleys.

    Each label's binary mask is blurred with a Gaussian of radius *sigma*
    pixels. Every fg pixel is then (re-)assigned to the label with the
    highest blurred score at that location, effectively rounding the
    boundary curves.

    Pixels that sit on a real intensity valley are **frozen**: they keep
    their original label and are excluded from reassignment. A narrow dilation
    band around each valley is also frozen so the smoothed boundary cannot
    erode into the valley.

    When *pp_smooth* is provided, frozen pixels are those where EITHER the PV
    signal is dark (high *inverted*) OR the PP signal is bright - the element-
    wise maximum of both channels is used.  This prevents boundary smoothing
    from sliding across portal tracts that are visible in PP but not
    (or weakly) visible in PV.

    Parameters
    ----------
    labels : np.ndarray
        Integer label map (0 = background, >0 = lobule id).
    inverted : np.ndarray
        Inverted PV stain: high where the PV signal is dark (inter-lobule
        boundary / valley region). Same spatial extent as *labels*.
    fg : np.ndarray
        Foreground mask (tissue minus vessel holes).
    sigma : float
        Gaussian smoothing radius in pixels. Larger -> rounder boundaries.
        0 disables the step entirely.
    valley_pct : float
        Percentile of the combined boundary-evidence distribution above which
        a pixel is frozen. E.g. 75 freezes the top 25 % (strongest boundary
        evidence).  Lower -> more pixels frozen (less smoothing but boundaries
        faithfully preserved).
    valley_band_px : int
        Dilation radius (px) applied around frozen pixels.  Prevents the
        smoothed boundary from creeping up to or across the valley.
    pp_smooth : np.ndarray or None
        Fine-scale smoothed PP stain (float32, [0, 1] in *fg*).  High values
        indicate portal-tract tissue that should be frozen regardless of PV.

    Returns
    -------
    np.ndarray
        Smoothed int32 label map. Background and vessel-hole pixels remain 0.
    """
    if sigma <= 0.0:
        return labels.astype(np.int32, copy=True)

    # Combine PV and PP boundary evidence: dark PV OR bright PP = lobule wall.
    boundary_evidence = (
        np.maximum(inverted, pp_smooth) if pp_smooth is not None else inverted
    )

    fg_vals = boundary_evidence[fg]
    if len(fg_vals) == 0:
        return labels.astype(np.int32, copy=True)

    valley_thr = float(np.percentile(fg_vals, valley_pct))

    # Build frozen mask: real valleys + a surrounding buffer band.
    valley_core = fg & (boundary_evidence > valley_thr)
    if valley_band_px > 0:
        frozen = binary_dilation(valley_core, iterations=valley_band_px) & fg
    else:
        frozen = valley_core

    H, W = labels.shape
    result = labels.copy().astype(np.int32)
    best_score = np.full((H, W), -np.inf, dtype=np.float32)

    for lid in np.unique(labels):
        if lid == 0:
            continue
        mask_f = (labels == lid).astype(np.float32)
        blurred = gaussian_filter(mask_f, sigma=sigma)
        update = ~frozen & fg & (blurred > best_score)
        best_score[update] = blurred[update]
        result[update] = lid

    result[~fg] = 0
    return result


def _smooth_contours(labels: np.ndarray, fg: np.ndarray,
                     sigma: float = 5.0) -> np.ndarray:
    """Smooth label boundaries via decimation + B-spline fitting.

    Two-stage smoothing per label contour:
    1. Decimate with Douglas-Peucker (epsilon = sigma * 0.5) to remove
       staircase vertices while preserving overall shape.
    2. Fit a periodic cubic B-spline with smoothing factor proportional to
       *perimeter* (not vertex count), so the smoothing strength is
       independent of how jagged the input is.

    The result is globally smooth curves whose frequency content matches
    biological lobule outlines (energy concentrated in the first ~10-20
    harmonics).
    """
    if sigma <= 0:
        return labels
    H, W = labels.shape
    out = np.zeros_like(labels)
    for lid in np.unique(labels):
        if lid <= 0:
            continue
        mask_u8 = (labels == lid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            perim = cv2.arcLength(cnt, closed=True)
            if perim < 30:
                cv2.drawContours(out, [cnt], -1, int(lid),
                                 thickness=cv2.FILLED)
                continue
            dp_eps = sigma * 0.5
            decimated = cv2.approxPolyDP(cnt, dp_eps, closed=True)
            pts = decimated.squeeze(1).astype(np.float64)
            if len(pts) < 6:
                cv2.drawContours(out, [cnt], -1, int(lid),
                                 thickness=cv2.FILLED)
                continue
            try:
                tck, _ = splprep([pts[:, 0], pts[:, 1]],
                                 s=perim * sigma * 0.1,
                                 per=True, k=3)
                n_eval = max(int(perim), 200)
                sx, sy = splev(np.linspace(0, 1, n_eval), tck)
            except Exception:
                cv2.drawContours(out, [cnt], -1, int(lid),
                                 thickness=cv2.FILLED)
                continue
            sx = np.clip(sx, 0, W - 1)
            sy = np.clip(sy, 0, H - 1)
            smoothed = np.stack([sx, sy], axis=1).astype(np.int32)
            cv2.drawContours(out, [smoothed.reshape(-1, 1, 2)], -1,
                             int(lid), thickness=cv2.FILLED)
    out[~fg] = 0

    # Fill pixels left bare by contour shrinkage.
    #
    # This used to be a single global Euclidean nearest-label lookup, the same
    # pattern documented as a measured regression in `_postprocess_labels`:
    # foreground islands that touch no label get adopted by whichever label is
    # nearest in a straight line, producing disconnected specks that each cost
    # a full 0.0 in `lobule_miou` under penalize_fp.  It matters here because
    # this path is live for any `contour_smooth_sigma > 0` rather than dormant.
    #
    # Same remedy as there: exclude foreground components that contain no
    # surviving label, so a distant label can never claim an island it does not
    # touch.  Unreachable islands are left at 0 rather than given a false owner.
    _orphan = np.zeros_like(fg, dtype=bool)
    _fgcc, _nfg = _ndlabel(fg)
    if _nfg > 0:
        _lab_per_comp = np.bincount(_fgcc[out > 0].ravel(), minlength=_nfg + 1)
        _orphan_ids = np.nonzero(_lab_per_comp[1:] == 0)[0] + 1
        if _orphan_ids.size:
            _orphan = np.isin(_fgcc, _orphan_ids)

    unlabelled = fg & (out == 0) & ~_orphan
    if unlabelled.any() and (out > 0).any():
        _, idx = _edt(out == 0, return_indices=True)
        out[unlabelled] = out[idx[0][unlabelled], idx[1][unlabelled]]
    out[_orphan] = 0
    return out


def _postprocess_labels(labels: np.ndarray, fg: np.ndarray, min_area: int = 1000) -> np.ndarray:
    """Zero labels outside *fg*, remove small regions, reassign their pixels to the nearest
    remaining label.

    Removing a small region and leaving its pixels as 0 silently discards foreground
    area and hurts mIoU. Instead we assign each removed pixel to the nearest
    surviving label via the distance transform, equivalent to a nearest-neighbour
    Voronoi fill on the remaining labels.

    Parameters
    ----------
    labels:
        Integer label map from watershed.
    fg:
        Boolean foreground mask.
    min_area:
        Minimum region size in pixels; smaller regions are removed and their
        pixels reassigned.

    Returns
    -------
    np.ndarray
        Cleaned int32 label map.
    """
    labels = labels.copy()
    labels[~fg] = 0
    _kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for lid in np.unique(labels[labels > 0]):
        _mask = labels == lid
        _opened = cv2.morphologyEx(_mask.astype(np.uint8), cv2.MORPH_OPEN, _kern).astype(bool)
        if _opened.sum() < min_area:
            labels[_mask] = 0
            continue
        _cc, _ncc = _ndlabel(_opened)
        for _cid in range(1, _ncc + 1):
            _comp = _cc == _cid
            if _comp.sum() < min_area:
                labels[_comp] = 0
        if _opened.sum() < _mask.sum():
            labels[_mask & ~_opened] = 0

    # Reassign unlabeled fg pixels to the nearest surviving label.
    #
    # This fill used to be a single global Euclidean nearest-label lookup,
    # which silently undid the area filter directly above it. Ragged tissue
    # borders leave small foreground islands that never received a watershed
    # label; the Euclidean fill adopted each one into whichever label happened
    # to be nearest in a straight line, often a lobule far away and not even
    # touching it, recreating exactly the disconnected specks the filter had
    # just removed. Those specks are expensive: lobule_miou runs with
    # penalize_fp=True, so every unmatched predicted region >= min_pred_px
    # contributes 0.0 to the mean.
    #
    # The fill is now geodesic: labels grow outward only into pixels they
    # actually touch, so an isolated island can never be captured by a distant
    # label.  Islands that no label reaches are left unlabelled rather than
    # force-assigned.
    # Identify foreground components that contain no surviving label at all -
    # these are the ragged border islands.  They must not take part in the fill,
    # because the nearest label by straight-line distance may be an unrelated
    # lobule they do not touch.
    _orphan = np.zeros_like(fg, dtype=bool)
    _fgcc, _nfg = _ndlabel(fg)
    if _nfg > 0:
        _lab_per_comp = np.bincount(_fgcc[labels > 0].ravel(), minlength=_nfg + 1)
        _orphan_ids = np.nonzero(_lab_per_comp[1:] == 0)[0] + 1
        if _orphan_ids.size:
            _orphan = np.isin(_fgcc, _orphan_ids)

    unlabeled = fg & (labels == 0) & ~_orphan
    if unlabeled.any() and (labels > 0).any():
        _, idx = _edt(labels == 0, return_indices=True)
        labels[unlabeled] = labels[idx[0][unlabeled], idx[1][unlabeled]]

    # Orphan islands stay unlabelled rather than being assigned a distant owner.
    labels[_orphan] = 0

    return labels.astype(np.int32)


def _merge_peakless_lobules(
    labels: np.ndarray,
    smoothed_blob: np.ndarray,
    fg: np.ndarray,
    peak_thresh_pct: float,
) -> tuple:
    """Merge lobules that contain no genuine PV-stain peak into their best neighbour.

    A lobule "has a peak" when its maximum blob-smoothed PV value (inside fg)
    exceeds the *peak_thresh_pct*-th percentile of all fg blob values. Peakless
    lobules are fragments (edge slivers, split artefacts) and are absorbed into
    whichever neighbour they share the most boundary pixels with.

    Merging is iterated until every surviving lobule contains a peak.

    A PP-based veto was tried here (skip the merge when a real portal tract sits
    on the shared boundary, on the theory that such a lobule is genuine but has
    its CV out of the section plane).  Measured on the full instance set it cost
    0.006 PP IoU: the pipeline already runs correct-to-over-segmented, so
    suppressing merges deepens over-segmentation rather than rescuing lobules.

    Returns
    -------
    labels : np.ndarray (int32)
        Updated label map.
    peak_coords : dict[int, tuple[int, int]]
        Surviving label id -> (row, col) of the brightest fg blob pixel in
        that label.  Used by the caller to re-seed the second-pass watershed.
    """
    labels = labels.copy().astype(np.int32)
    H, W = labels.shape

    # blob_fg: smoothed PV restricted to fg (zero outside)
    blob_fg = (smoothed_blob * fg.astype(np.float32)).astype(np.float32)
    fg_vals = blob_fg[blob_fg > 0]
    thr = float(np.percentile(fg_vals, peak_thresh_pct)) if len(fg_vals) else 0.0

    def _peaks_per_label():
        """Return {lid: (max_val, row, col)} for all positive labels."""
        result = {}
        for lid in np.unique(labels[labels > 0]):
            lid = int(lid)
            ys, xs = np.where(fg & (labels == lid))
            if len(ys) == 0:
                result[lid] = (0.0, 0, 0)
                continue
            vals = blob_fg[ys, xs]
            best = int(np.argmax(vals))
            result[lid] = (float(vals[best]), int(ys[best]), int(xs[best]))
        return result

    def _bnd_counts():
        """Count shared boundary pixels for every adjacent label pair."""
        counts = defaultdict(lambda: defaultdict(int))
        max_id = int(labels.max()) + 1
        for dy, dx in ((0, 1), (1, 0)):
            la = labels[: H - dy if dy else H, : W - dx if dx else W]
            lb = labels[dy:, dx:]
            bnd = (la != lb) & (la > 0) & (lb > 0)
            if not bnd.any():
                continue
            la_b = la[bnd].astype(np.int64)
            lb_b = lb[bnd].astype(np.int64)
            codes, cnts = np.unique(la_b * max_id + lb_b, return_counts=True)
            for code, cnt in zip(codes.tolist(), cnts.tolist()):
                a, b = int(code // max_id), int(code % max_id)
                counts[a][b] += int(cnt)
                counts[b][a] += int(cnt)
        return counts

    # Iterative merge until stable
    for _ in range(50):                         # safety cap
        info = _peaks_per_label()
        peakless = [l for l, (mv, _, _) in info.items() if mv < thr]
        if not peakless:
            break
        counts = _bnd_counts()
        changed = False
        for lid in peakless:
            if not (labels == lid).any():
                continue                         # absorbed in this round
            nb = counts.get(lid)
            if not nb:
                continue                         # isolated -> left for postprocess
            best_nb = max(nb, key=nb.__getitem__)
            labels[labels == lid] = best_nb
            changed = True
        if not changed:
            break

    # Collect peak positions for surviving labels
    peak_coords: dict = {}
    for lid, (_, py, px) in _peaks_per_label().items():
        peak_coords[lid] = (py, px)

    return labels, peak_coords


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_lobules(
    tissue: np.ndarray,
    vessel_holes: np.ndarray,
    pv_stain: np.ndarray,
    *,
    cv_mask: np.ndarray | None = None,
    pp_stain: np.ndarray | None = None,
    blob_sigma: float = 50.0,
    peak_min_dist: int = 220,
    peak_thresh_pct: float = 40.0,
    valley_sigma: float = 25.0,
    cluster_merge_sigma: float = 0.0,
    min_area_px: int = 20000,
    valley_merge_thr_pct: float | None = None,
    boundary_smooth_sigma: float = 31.0,
    boundary_smooth_valley_pct: float = 51.5,
    boundary_smooth_valley_band_px: int = 14,
    seed_tissue_ring: int = 8,
    pp_thr: float | None = None,
    pp_blend: float = 0.0,
    fused_merge_pv_pct: float | None = None,
    valley_merge_sigma: float = 8.0,
    contour_smooth_sigma: float = 0.0,
    edge_grad_blend: float = 0.7,
    edge_local_contrast_sigma: float = 0.0,
    min_seed_border_dist: int = 0,
    peak_thresh_factor: float = 0.0,
    seed_points_xy=None,
    seed_point_radius: int = 0,
    return_debug: bool = False,
) -> np.ndarray:
    """Segment liver lobules via Higra seeded watershed.

    **Seeding** (when *cv_mask* is provided):
        One seed is placed at the centroid of every connected central-vein (CV)
        component.  Intensity peaks (local maxima in the smoothed PV channel)
        are added as fallback seeds only for tissue regions not already covered
        by a CV seed, suppressed within *peak_min_dist* of any CV pixel.
        Without *cv_mask*, only intensity peaks are used (original behaviour).

    **Watershed** runs on an inverted masked-Gaussian-smoothed PV field so that
        dark inter-lobule boundaries (low PV signal) form high-cost edges.

    **Valley merge** (automatic, always active):
        After the watershed, adjacent lobule pairs whose shared boundary never
        drops below an Otsu-derived valley threshold are merged.  This corrects
        over-segmentation caused by spurious seeds in regions without a real
        dark portal-tract boundary.

    Parameters
    ----------
    tissue:
        Boolean mask of foreground tissue (full image extent).
    vessel_holes:
        Boolean mask of vessel interiors to exclude from seeding.
    pv_stain:
        Float32 PV-channel intensity image (same spatial extent as *tissue*).
    cv_mask : np.ndarray or None
        Boolean mask of central vein lumens.  When provided each connected
        component gets exactly one seed; intensity peaks are used only for
        uncovered tissue.  Pass ``None`` to use intensity peaks exclusively
        (backward-compatible default).
    pp_stain : np.ndarray or None
        Float32 PP-channel intensity image (same extent as *tissue*).  PP is
        brightest at the inter-lobule boundary, so the smoothed PP field is
        directly usable as a watershed edge-cost map.  Has no effect unless
        *pp_blend* > 0.  When ``None`` only the inverted PV is used.
    pp_blend : float
        Weight of the PP field in the edge map:
        ``edge_map = (1 - pp_blend) * inverted_pv + pp_blend * pp_smooth``.
        0 disables PP entirely (backward-compatible default), 1 makes PP the
        sole boundary evidence.  Worth a real weight because the PV zonation
        curve is flat at the boundary while the PP curve is not - see the
        comment at the blend site.
    blob_sigma:
        Gaussian sigma for fine-scale peak detection (approx. lobule radius).
    peak_min_dist:
        Minimum pixel distance between kept peaks (NMS radius).
    peak_thresh_pct:
        Percentile brightness threshold for interior peaks.
    valley_sigma:
        Gaussian sigma for the valley-smoothed field used as watershed weights.
        Larger values produce smoother, more rounded lobule boundaries.
    cluster_merge_sigma:
        Coarse-scale sigma for basin-merging. 0 or None disables merging.
    min_area_px:
        Minimum lobule area in pixels; smaller labelled regions are zeroed out.
    boundary_smooth_sigma : float
        Gaussian radius (pixels) for post-watershed boundary smoothing.
        Each label's binary mask is blurred and pixels are reassigned to the
        winning label, producing geometrically smooth boundaries. Valley
        pixels are frozen so the smoothing never crosses a real intensity
        valley. Set to 0 to disable.
    boundary_smooth_valley_pct : float
        Percentile of the inverted-PV distribution above which a pixel is
        considered a real valley and kept frozen during smoothing. 75 means
        the darkest 25 % of foreground pixels (the actual inter-lobule
        boundaries) are untouched; the rest can be reassigned to smooth the
        boundary curves.
    boundary_smooth_valley_band_px : int
        Dilation radius (pixels) applied around frozen valley pixels.
        Prevents the smoothed boundary from creeping up to or across a
        valley. Set to 0 to disable the buffer band.

    Returns
    -------
    np.ndarray
        Integer (int32) label map of shape ``tissue.shape``. Background and
        vessel-hole pixels are labelled 0; each lobule gets a unique label
        starting from 1.

    Raises
    ------
    ImportError
        If the ``higra`` package is not installed.
    """
    try:
        import higra as hg
    except ImportError as exc:
        raise ImportError(
            "The 'higra' package is required for segment_lobules. "
            "Install it with: pip install higra"
        ) from exc

    if blob_sigma <= 0:
        raise ValueError(f"blob_sigma must be positive, got {blob_sigma}")
    if peak_min_dist < 1:
        raise ValueError(f"peak_min_dist must be >= 1, got {peak_min_dist}")
    if not (0 <= peak_thresh_pct <= 100):
        raise ValueError(f"peak_thresh_pct must be in [0, 100], got {peak_thresh_pct}")

    fg = tissue & ~vessel_holes
    H, W = tissue.shape

    # Blob-scale PV smooth: reused for seeding, peak validation, and re-seeding.
    smoothed_blob = _masked_smooth(pv_stain, fg, blob_sigma)

    # -- Seeding --------------------------------------------------------------
    # An explicit seed set replaces the detectors entirely. This is what the
    # interactive preview passes back after the user has edited the seed points:
    # the layer is pre-filled with the detected seeds, so "edited" already
    # includes everything the detectors found unless the user removed it.
    if seed_points_xy is not None and len(seed_points_xy):
        _pts = np.asarray(seed_points_xy, float).reshape(-1, 2)
        _r = int(seed_point_radius) if seed_point_radius > 0 else             max(3, int(round(0.02 * max(H, W))))
        seed_cnts = []
        for _x, _y in _pts:
            _xi, _yi = int(round(_x)), int(round(_y))
            if not (0 <= _xi < W and 0 <= _yi < H):
                continue
            _disc = np.zeros((H, W), np.uint8)
            cv2.circle(_disc, (_xi, _yi), _r, 1, cv2.FILLED)
            _c, _ = cv2.findContours(_disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if _c:
                seed_cnts.append(_c[0])
        cv_seed_cnts, peak_seed_cnts = seed_cnts, []
    elif cv_mask is not None and cv_mask.any():
        # One seed per CV component; exclude a neighbourhood around each CV
        # from intensity-peak detection so peaks don't duplicate CV seeds.
        # Distance transform is O(H*W) regardless of peak_min_dist, avoiding
        # the massive kernel that cv2.dilate would need at large peak_min_dist.
        cv_seed_cnts = _seeds_from_cv_mask(cv_mask, tissue,
                                                  seed_tissue_ring=seed_tissue_ring,
                                                  min_border_dist=min_seed_border_dist)
        _cv_dist = _edt(~cv_mask).astype(np.float32)
        existing = (_cv_dist <= peak_min_dist)
        peak_seed_cnts = _find_intensity_cluster_seeds(
            pv_stain, fg, existing,
            blob_sigma=blob_sigma,
            peak_min_dist=peak_min_dist,
            peak_thresh_pct=peak_thresh_pct,
            cluster_merge_sigma=cluster_merge_sigma if cluster_merge_sigma and cluster_merge_sigma > 0 else None,
            cv_dist_map=_cv_dist,
            min_seed_border_dist=min_seed_border_dist,
            peak_thresh_factor=peak_thresh_factor,
        )
        seed_cnts = cv_seed_cnts + peak_seed_cnts
    elif True:
        cv_seed_cnts = []
        peak_seed_cnts = _find_intensity_cluster_seeds(
            pv_stain, fg, np.zeros((H, W), dtype=bool),
            blob_sigma=blob_sigma,
            peak_min_dist=peak_min_dist,
            peak_thresh_pct=peak_thresh_pct,
            cluster_merge_sigma=cluster_merge_sigma if cluster_merge_sigma and cluster_merge_sigma > 0 else None,
            min_seed_border_dist=min_seed_border_dist,
            peak_thresh_factor=peak_thresh_factor,
        )
        seed_cnts = peak_seed_cnts

    # Seed centroids (x, y) for diagnostics, split by origin so callers can
    # visualise intensity-peak seeds separately from CV-derived seeds.
    cv_seed_xy   = _seed_centroids(cv_seed_cnts)
    peak_seed_xy = _seed_centroids(peak_seed_cnts)

    if not seed_cnts:
        if return_debug:
            return np.zeros((H, W), np.int32), {
                "cv_seed_xy": cv_seed_xy,
                "peak_seed_xy": peak_seed_xy,
            }
        return np.zeros((H, W), np.int32)

    markers = _make_markers_from_cnts(seed_cnts, H, W)

    smoothed = _masked_smooth(pv_stain, tissue, valley_sigma)

    mx_val = smoothed[tissue].max() if tissue.any() else 1.0
    inverted = mx_val - smoothed
    inverted[~tissue] = mx_val

    # -- Gradient magnitude edge map ------------------------------------------
    # Local contrast normalization equalizes per-lobule expression differences
    # so that Sobel gradient detects all boundaries, even between a bright and
    # dim lobule where the absolute gradient would be near zero.
    if edge_local_contrast_sigma > 0:
        _lcn_mean = _masked_smooth(pv_stain, tissue, edge_local_contrast_sigma)
        _lcn_sq = _masked_smooth(pv_stain ** 2, tissue, edge_local_contrast_sigma)
        _lcn_std = np.sqrt(np.maximum(_lcn_sq - _lcn_mean ** 2, 1e-6))
        _pv_lcn = (pv_stain - _lcn_mean) / _lcn_std
        _pv_lcn[~tissue] = 0.0
        smoothed_grad = _masked_smooth(_pv_lcn, tissue, valley_sigma)
    else:
        smoothed_grad = smoothed
    gx = sobel(smoothed_grad, axis=1)
    gy = sobel(smoothed_grad, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag[~tissue] = 0.0
    _gmax = grad_mag[tissue].max() if tissue.any() else 1.0
    grad_norm = (grad_mag / _gmax) if _gmax > 0 else grad_mag
    _inv_norm = inverted / max(mx_val, 1e-6)

    # Blend: edge_grad_blend=1 → pure gradient, 0 → pure inverted (legacy).
    inverted_pv = edge_grad_blend * grad_norm + (1.0 - edge_grad_blend) * _inv_norm
    inverted_pv[~tissue] = 1.0

    # -- PP edge evidence -----------------------------------------------------
    _pp_sm   = None
    _pp_fine = None
    _pp_thr  = 0.5
    if pp_stain is not None:
        _pp_sm = _masked_smooth(pp_stain, tissue, min(valley_sigma, 20.0))
        _pp_sm[~tissue] = 1.0
        _pp_fine = _masked_smooth(pp_stain, tissue, 8.0)
        _pp_fine[~tissue] = 0.0
        if pp_thr is not None:
            _pp_vals = _pp_fine[fg]
            if _pp_vals.size > 0:
                _pp_thr = float(np.percentile(_pp_vals, pp_thr))
            else:
                _pp_thr = 0.5
        else:
            try:
                _pp_thr = float(threshold_otsu(_pp_fine[fg])) if fg.any() else 0.5
            except Exception:
                _pp_thr = 0.5

        # PP is bright at the lobule boundary, so the smoothed PP field is
        # already an edge-cost map in the same [0, 1] units as inverted_pv; no
        # transform is needed, only a proportional blend (same pattern as
        # edge_grad_blend above).
        #
        # This matters because the PV zonation curve is cubic-flat at the
        # boundary (hill n=3 -> zero derivative at P=0, ~0.07% change over
        # +/-5 px) whereas the PP curve is exponential with a non-zero slope
        # there (~7.9% over the same span).  Practically all of the boundary
        # signal lives in PP.
        #
        # Earlier attempts combined the two with np.maximum, which is
        # winner-take-all between fields whose magnitudes are not calibrated
        # against each other, and with gradient / skeleton transforms that
        # place their response away from the actual boundary.  All measured
        # negative; see the module history.
        if pp_blend > 0.0:
            edge_map = (1.0 - pp_blend) * inverted_pv + pp_blend * _pp_sm
        else:
            edge_map = inverted_pv
    else:
        edge_map = inverted_pv

    edge_map = cv2.medianBlur(
        edge_map.astype(np.float32), 3,
    )

    graph = hg.get_8_adjacency_graph((H, W))
    edge_weights = hg.weight_graph(graph, edge_map, hg.WeightFunction.mean)
    labels = hg.labelisation_seeded_watershed(graph, edge_weights, markers)

    # Per-stage lobule counts, collected only when return_debug is set.  Each
    # stage below can merge or drop labels; recording the count after every one
    # localises where lobules are lost relative to the seed set.
    _stage_counts: list = []

    def _tally(_name: str) -> None:
        if return_debug:
            _stage_counts.append(
                (_name, int(len(set(labels[fg].ravel()) - {0})))
            )

    _tally("seeded_watershed")

    # -- Valley merge: join over-segmented pairs with no dark boundary -----
    # Use a separate, small-sigma PV smooth (valley_merge_sigma) so that real
    # dark valleys are not blurred away at large valley_sigma values.
    #
    # Protected labels (CV-seeded) are normally never merged. Exception: when
    # PP is also dark on the shared boundary, both channels agree there is no
    # portal tract, so the pair is a fused super-lobule and should be merged.
    _protected: frozenset = frozenset()
    if cv_mask is not None and cv_mask.any():
        _cv_dil1 = cv2.dilate(
            cv_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        _protected = frozenset(
            int(l) for l in np.unique(labels[_cv_dil1 & fg]) if l > 0
        )

    if fg.any():
        _smooth_merge = _masked_smooth(pv_stain, tissue, valley_merge_sigma)
        try:
            # Percentile threshold is more robust than Otsu under residual
            # shading: it always selects the darkest N% of foreground pixels
            # as valleys regardless of global intensity level.
            if valley_merge_thr_pct is not None:
                valley_thr = float(np.percentile(_smooth_merge[fg], valley_merge_thr_pct))
            else:
                valley_thr = float(threshold_otsu(_smooth_merge[fg]))
            # Stricter PV gate for CV-seeded (protected) pairs: only merge when
            # the seam is brighter than fused_merge_pv_pct of fg pixels (>= a
            # high percentile of the smoothed PV).  Fused super-lobules have a
            # bright, valley-less seam; genuinely separate lobules keep their
            # dark periportal valley and are excluded.  None -> legacy behaviour.
            if fused_merge_pv_pct is not None:
                _prot_pv_thr = float(np.percentile(_smooth_merge[fg], fused_merge_pv_pct))
            else:
                _prot_pv_thr = None
            labels = _merge_below_valley(
                labels, _smooth_merge, fg, valley_thr,
                protected_labels=_protected,
                pp_smooth=_pp_fine,
                pp_thr=_pp_thr,
                protected_merge_pv_thr=_prot_pv_thr,
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Valley merge skipped (%s)", exc,
            )
    _tally("valley_merge")

    labels = _postprocess_labels(labels, fg, min_area=min_area_px)
    _tally("postprocess_min_area")

    # -- Peak validation: merge peakless lobules -------------------------------
    # Each genuine lobule has a PV brightness peak (perivenular zone near the CV).
    # Lobules without a peak are fragments: split artefacts or edge slivers.
    # They are absorbed into the neighbour with the longest shared boundary.
    labels, peak_coords = _merge_peakless_lobules(
        labels, smoothed_blob, fg, peak_thresh_pct,
    )
    _tally("merge_peakless")

    # -- Fine-scale peak refinement --------------------------------------------
    # _merge_peakless_lobules returns the peak at blob_sigma scale ( approx.  lobule
    # radius), which can be offset by several pixels from the true CV centre.
    # Re-localise each peak at a finer Gaussian scale (30 % of blob_sigma) so
    # that the second-pass watershed seeds are placed as close as possible to
    # the actual PV brightness maximum (i.e. the CV lumen edge).
    if peak_coords:
        _fine_sigma = max(4.0, blob_sigma * 0.3)
        _smoothed_fine = _masked_smooth(pv_stain, fg, _fine_sigma)
        _refined_coords: dict = {}
        for _lid, (_py, _px) in peak_coords.items():
            _ys, _xs = np.where(fg & (labels == _lid))
            if len(_ys) == 0:
                _refined_coords[_lid] = (_py, _px)
                continue
            _vals = _smoothed_fine[_ys, _xs]
            _best = int(np.argmax(_vals))
            _refined_coords[_lid] = (int(_ys[_best]), int(_xs[_best]))
        peak_coords = _refined_coords

    # -- Second-pass watershed from confirmed PV peaks -------------------------
    # After the merge the seed set has changed. Re-seed from the refined PV
    # brightness maxima (inside fg tissue) rather than from the original CV
    # centroids. The same PV+PP edge-weight graph is reused, with no extra
    # computation. Every fg pixel is freshly assigned to its nearest confirmed
    # lobule centre through the strongest available boundary landscape.
    if peak_coords:
        _fresh = np.zeros((H, W), np.int32)
        _r = max(5, peak_min_dist // 8)
        for _lid, (_py, _px) in peak_coords.items():
            cv2.circle(_fresh, (int(_px), int(_py)), _r, int(_lid), -1)
        labels = hg.labelisation_seeded_watershed(graph, edge_weights, _fresh)
        labels[~fg] = 0
        labels = labels.astype(np.int32)
    _tally("second_watershed")

    # -- CV-aware spurious-fragment removal -----------------------------------
    # After the second-pass watershed, small regions that contain no CV pixel
    # are likely boundary artefacts or tissue-edge slivers. Absorb them into
    # the nearest surviving neighbour (Voronoi fill). Regions that do contain
    # a CV are kept unconditionally; removing them would discard a genuine
    # lobule seed and leave a gap in the portality map.
    if cv_mask is not None and cv_mask.any():
        _second_ids = np.unique(labels[labels > 0])
        _removed = False
        for _lid in _second_ids:
            _rgn = labels == _lid
            if not (cv_mask & _rgn).any() and int(_rgn.sum()) < min_area_px:
                labels[_rgn] = 0
                _removed = True
        if _removed:
            _unlab = fg & (labels == 0)
            if _unlab.any() and (labels > 0).any():
                _, _idx = _edt(labels == 0, return_indices=True)
                labels[_unlab] = labels[_idx[0][_unlab], _idx[1][_unlab]]
            labels = labels.astype(np.int32)
    _tally("cv_aware_fragment_removal")

    if boundary_smooth_sigma > 0:
        labels = _smooth_labels_valley_aware(
            labels, inverted, fg,
            sigma=boundary_smooth_sigma,
            valley_pct=boundary_smooth_valley_pct,
            valley_band_px=boundary_smooth_valley_band_px,
            pp_smooth=None,
        )
    _tally("boundary_smooth")

    if contour_smooth_sigma > 0:
        labels = _smooth_contours(labels, fg, sigma=contour_smooth_sigma)
    _tally("contour_smooth")

    # Boundary smoothing can create tiny disconnected slivers, so a final pass.
    labels = _postprocess_labels(labels, fg, min_area=min_area_px)
    _tally("final_postprocess")

    if return_debug:
        return labels, {
            "cv_seed_xy": cv_seed_xy,
            "peak_seed_xy": peak_seed_xy,
            "n_seeds": int(len(seed_cnts)),
            "stage_counts": _stage_counts,
        }
    return labels
