"""Higra seeded watershed segmentation of liver lobules.

Public API
----------
segment_lobules(tissue, vessel_holes, pv_stain, *, ...)
    Main entry point. Returns an integer label map (np.int32) of the same
    spatial extent as *tissue*.

All other names in this module are private helpers (underscore prefix).
"""

from __future__ import annotations

import numpy as np
import cv2
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt as _edt
from skimage.filters import threshold_otsu


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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
    mx = smoothed[mask].max() if mask.any() else 1.0
    return (smoothed / mx) if mx > 0 else smoothed


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
        m = np.zeros((H, W), np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
        markers[m > 0] = i
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
        Minimum pixel distance between kept peaks (NMS radius). Interior peaks
        use the full distance; edge peaks use half.
    seed_radius:
        Radius of the circular seed contour painted around each peak.
        Defaults to ``max(10, peak_min_dist // 4)``.
    peak_thresh_pct:
        Percentile brightness threshold for interior peaks. Edge peaks use
        half this threshold (minimum 5th percentile).
    tissue_boundary_dist:
        Precomputed distance-transform-from-tissue-boundary array. Computed
        internally when *None*.
    cluster_merge_sigma:
        Gaussian sigma for the coarse-scale field used to define lobule basins.
        Should be large enough to blur over vessel holes (typically 2–4 times
        *blob_sigma*). When *None* or ≤ 0, cluster merging is disabled. Fine-
        scale peaks that map to the same coarse-scale local maximum (within
        *peak_min_dist*) are merged to the single brightest fine peak.

    Returns
    -------
    list of np.ndarray
        Circular seed contours (one per kept peak) in OpenCV format.
    """
    from scipy.ndimage import maximum_filter
    from scipy.ndimage import gaussian_filter as gf

    H, W = pv_stain.shape
    if seed_radius is None:
        seed_radius = max(10, peak_min_dist // 4)

    if tissue_boundary_dist is None:
        tissue_boundary_dist = _edt(fg).astype(np.float32)

    # Step 1: masked Gaussian smooth for fine-scale peak detection
    pv_f = pv_stain.astype(np.float32).copy()
    pv_f[~fg] = 0.0
    mask_f = fg.astype(np.float32)
    num = gf(pv_f, sigma=blob_sigma)
    den = np.maximum(gf(mask_f, sigma=blob_sigma), 1e-6)
    blurred = num / den
    blurred[~fg] = 0.0

    # Step 2: find local maxima at two scales
    edge_zone = tissue_boundary_dist < peak_min_dist

    # Optional: CV-distance info for zone-aware threshold and combined NMS score.
    # cv_far = fg pixels far from any CV and not in the tissue edge zone - the
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

    local_max_full = maximum_filter(blurred, size=peak_min_dist)
    local_max_half = maximum_filter(blurred, size=max(3, peak_min_dist // 2))
    peaks_interior = (blurred == local_max_full) & (blurred > 0) & fg & ~edge_zone
    peaks_edge = (blurred == local_max_half) & (blurred > 0) & fg & edge_zone
    peaks = peaks_interior | peaks_edge

    fg_vals = blurred[fg & (blurred > 0)]
    if len(fg_vals) == 0:
        return []
    thr_interior = np.percentile(fg_vals, peak_thresh_pct)
    thr_edge = np.percentile(fg_vals, max(5.0, peak_thresh_pct * 0.5))

    if cv_dist_map is not None and cv_far is not None:
        # Relax threshold for uncovered territory: CV-dropout lobules have weak PV
        # expression and their peaks would be cut by the normal interior threshold.
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

    # Step 3: NMS - sort by combined score, suppress nearby duplicates.
    # When cv_dist_map is available, weight brightness by distance from the
    # nearest CV so that a well-centred (far-from-CV) peak beats a slightly
    # brighter but CV-adjacent peak - better coverage of unclaimed territory.
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
        pv_c = pv_stain.astype(np.float32).copy()
        pv_c[~fg] = 0.0
        num_c = gf(pv_c, sigma=cluster_merge_sigma)
        den_c = np.maximum(gf(mask_f, sigma=cluster_merge_sigma), 1e-6)
        coarse = num_c / den_c
        coarse[~fg] = 0.0

        from scipy.ndimage import maximum_filter as mf
        coarse_maxfilt = mf(coarse, size=peak_min_dist)
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
) -> list:
    """One circular seed per connected CV component.

    The seed circle is centred at the component centroid and sized so that its
    outer ring always extends *seed_tissue_ring* pixels into the surrounding
    tissue, regardless of CV size.  This guarantees that some marker pixels are
    in the foreground (tissue & ~vessel_holes), giving the watershed a proper
    low-inverted-PV starting point - identical in spirit to the intensity-peak
    seeds that worked well before.  Seeding entirely inside the vessel lumen
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

    Returns
    -------
    list of np.ndarray
        Circular seed contours in OpenCV (N, 1, 2) int32 format, one per CV component.
    """
    from scipy.ndimage import label as ndlabel

    H, W = cv_mask.shape
    labeled_cv, n_cv = ndlabel(cv_mask)

    seed_cnts: list = []
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)

    for comp_id in range(1, n_cv + 1):
        ys = np.where(labeled_cv == comp_id)[0]
        if len(ys) < min_cv_area:               # skip noise pixels / tiny artefacts
            continue

        # Inscribed-circle centre via distance-transform maximum.
        # For a perfect circle this equals the centroid; for elongated or
        # irregular lumens it is the most interior point - geometrically the
        # true centre of the lumen, which is what the watershed needs as its
        # anchor so that portality flows outward from the correct origin.
        comp_mask = (labeled_cv == comp_id)
        _dt_cv = _edt(comp_mask)
        cy, cx = divmod(int(np.argmax(_dt_cv)), W)

        # Inscribed-circle radius (EDT value at the centre) is more accurate
        # than the area-based approximation, especially for elongated CVs.
        comp_radius = max(1, int(np.ceil(float(_dt_cv[cy, cx]))))
        seed_radius = comp_radius + seed_tissue_ring

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
) -> np.ndarray:
    """Merge adjacent lobule pairs that share no real PV valley on their boundary.

    Two adjacent labelled regions A and B are merged when the minimum smoothed-PV
    value along their shared pixel interface is >= *valley_thr* - meaning no dark
    inter-lobule region separates them, so they are most likely one lobule that was
    over-segmented.

    **Protected pairs** (both labels contain a CV): protected labels are normally
    skipped to avoid merging correctly-seeded lobules.  Exception: when *pp_smooth*
    is provided and the maximum PP signal along the shared boundary is below
    *pp_thr*, both PV and PP agree there is no portal tract - the pair is a fused
    super-lobule and is merged.  This catches the synthetic fusion case where two
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
        pixel.  Protected pairs whose max boundary PP >= *pp_thr* are NOT
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
    pair_max_pp: dict[int, float] = {}   # max PP on boundary per pair (portal-tract evidence)

    for dy, dx in ((0, 1), (1, 0)):
        la  = labels  [: H - dy if dy else H, : W - dx if dx else W]
        lb  = labels  [dy:, dx:]
        pv_a = smoothed_pv[: H - dy if dy else H, : W - dx if dx else W]
        pv_b = smoothed_pv[dy:, dx:]

        bnd = (la != lb) & (la > 0) & (lb > 0)
        if not bnd.any():
            continue

        la_b   = la[bnd].astype(np.int64)
        lb_b   = lb[bnd].astype(np.int64)
        # Conservative: minimum PV on either side of the interface
        pv_bnd = np.minimum(pv_a[bnd], pv_b[bnd])

        # PP: maximum on either side (high = portal tract present)
        pp_bnd = None
        if pp_smooth is not None:
            pp_a = pp_smooth[: H - dy if dy else H, : W - dx if dx else W]
            pp_b = pp_smooth[dy:, dx:]
            pp_bnd = np.maximum(pp_a[bnd], pp_b[bnd])

        # Canonicalise pair (a <= b) -> unique int64 code
        swap          = la_b > lb_b
        la_b[swap], lb_b[swap] = lb_b[swap].copy(), la_b[swap].copy()
        codes         = la_b * max_id + lb_b

        unique_codes, inv = np.unique(codes, return_inverse=True)
        min_pvs = np.full(len(unique_codes), np.inf, dtype=np.float64)
        np.minimum.at(min_pvs, inv, pv_bnd)

        max_pps = None
        if pp_bnd is not None:
            max_pps = np.full(len(unique_codes), -np.inf, dtype=np.float64)
            np.maximum.at(max_pps, inv, pp_bnd)

        for i, (code, min_pv) in enumerate(zip(unique_codes.tolist(), min_pvs.tolist())):
            code = int(code)
            if code in pair_min:
                if min_pv < pair_min[code]:
                    pair_min[code] = float(min_pv)
            else:
                pair_min[code] = float(min_pv)

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
        if min_pv >= valley_thr:
            a = int(code // max_id)
            b = int(code % max_id)
            if protected_labels and (a in protected_labels or b in protected_labels):
                # Protected (CV-seeded) pair: only merge when PP also confirms
                # absence of a portal tract -> fused super-lobule.
                if pp_smooth is not None:
                    max_pp = pair_max_pp.get(code, 1.0)
                    if max_pp >= pp_thr:
                        continue   # PP-bright boundary -> real portal tract -> keep separate
                    # Both PV (no valley) and PP (no portal tract) agree -> merge
                else:
                    continue   # no PP channel -> keep original protection
            if a in parent and b in parent:
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
    highest blurred score at that location - effectively rounding the
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

    from scipy.ndimage import binary_dilation

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


def _postprocess_labels(labels: np.ndarray, fg: np.ndarray, min_area: int = 1000) -> np.ndarray:
    """Zero labels outside *fg*, remove small regions, reassign their pixels to the nearest
    remaining label.

    Removing a small region and leaving its pixels as 0 silently discards foreground
    area and hurts mIoU.  Instead we assign each removed pixel to the nearest
    surviving label via the distance transform - equivalent to a nearest-neighbour
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
    for lid in np.unique(labels[labels > 0]):
        if (labels == lid).sum() < min_area:
            labels[labels == lid] = 0

    # Reassign unlabeled fg pixels to the nearest surviving label.
    unlabeled = fg & (labels == 0)
    if unlabeled.any() and (labels > 0).any():
        _, idx = _edt(labels == 0, return_indices=True)
        labels[unlabeled] = labels[idx[0][unlabeled], idx[1][unlabeled]]

    return labels.astype(np.int32)


def _merge_peakless_lobules(
    labels: np.ndarray,
    smoothed_blob: np.ndarray,
    fg: np.ndarray,
    peak_thresh_pct: float,
) -> tuple:
    """Merge lobules that contain no genuine PV-stain peak into their best neighbour.

    A lobule "has a peak" when its maximum blob-smoothed PV value (inside fg)
    exceeds the *peak_thresh_pct*-th percentile of all fg blob values.  Peakless
    lobules are fragments - edge slivers, split artefacts - and are absorbed into
    whichever neighbour they share the most boundary pixels with.

    Merging is iterated until every surviving lobule contains a peak.

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
) -> np.ndarray:
    """Segment liver lobules via Higra seeded watershed.

    **Seeding** (when *cv_mask* is provided):
        One seed is placed at the centroid of every connected central-vein (CV)
        component.  Intensity peaks (local maxima in the smoothed PV channel)
        are added as fallback seeds only for tissue regions not already covered
        by a CV seed - suppressed within *peak_min_dist* of any CV pixel.
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
        Float32 PP-channel intensity image (same extent as *tissue*).
        Portal tracts stain brightly for PP - combining this with inverted PV
        reinforces inter-lobule boundary edges in the watershed weight map and
        significantly sharpens boundary placement.  When ``None`` only the
        inverted PV is used (backward-compatible default).
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

    fg = tissue & ~vessel_holes
    H, W = tissue.shape

    # Blob-scale PV smooth: reused for seeding, peak validation, and re-seeding.
    smoothed_blob = _masked_smooth(pv_stain, fg, blob_sigma)

    # ── Seeding ──────────────────────────────────────────────────────────────
    if cv_mask is not None and cv_mask.any():
        # One seed per CV component; exclude a neighbourhood around each CV
        # from intensity-peak detection so peaks don't duplicate CV seeds.
        # Distance transform is O(H*W) regardless of peak_min_dist - avoids
        # the massive kernel that cv2.dilate would need at large peak_min_dist.
        cv_seed_cnts = _seeds_from_cv_mask(cv_mask, tissue)
        _cv_dist = _edt(~cv_mask).astype(np.float32)
        existing = (_cv_dist <= peak_min_dist)
        peak_seed_cnts = _find_intensity_cluster_seeds(
            pv_stain, fg, existing,
            blob_sigma=blob_sigma,
            peak_min_dist=peak_min_dist,
            peak_thresh_pct=peak_thresh_pct,
            cluster_merge_sigma=cluster_merge_sigma if cluster_merge_sigma and cluster_merge_sigma > 0 else None,
            cv_dist_map=_cv_dist,
        )
        seed_cnts = cv_seed_cnts + peak_seed_cnts
    else:
        seed_cnts = _find_intensity_cluster_seeds(
            pv_stain, fg, np.zeros((H, W), dtype=bool),
            blob_sigma=blob_sigma,
            peak_min_dist=peak_min_dist,
            peak_thresh_pct=peak_thresh_pct,
            cluster_merge_sigma=cluster_merge_sigma if cluster_merge_sigma and cluster_merge_sigma > 0 else None,
        )

    if not seed_cnts:
        return np.zeros((H, W), np.int32)

    markers = _make_markers_from_cnts(seed_cnts, H, W)

    smoothed = _masked_smooth(pv_stain, tissue, valley_sigma)

    mx_val = smoothed[tissue].max() if tissue.any() else 1.0
    inverted = mx_val - smoothed
    inverted[~tissue] = mx_val

    # ── Edge weights ─────────────────────────────────────────────────────────
    # Portal tracts are dark in PV (-> high inverted_pv) AND bright in PP.
    # Taking the element-wise maximum of both normalised signals gives the
    # strongest available boundary evidence at every pixel.  PP is smoothed at
    # a capped sigma (≤ 20 px) so that narrow portal-tract peaks survive even
    # when valley_sigma is large.
    #
    # A fine-scale (8 px) PP smooth is also kept separately for the valley-merge
    # and boundary-freeze steps, where we need per-pixel portal-tract evidence
    # rather than a spatially averaged field.
    _pp_sm   = None   # edge-weight PP (coarser, for watershed graph)
    _pp_fine = None   # fine-scale PP (for valley merge & boundary freeze)
    _pp_thr  = 0.5    # Otsu-derived PP threshold for merge decisions
    if pp_stain is not None:
        _pp_sm = _masked_smooth(pp_stain, tissue, min(valley_sigma, 20.0))
        _pp_sm[~tissue] = 1.0          # treat outside-tissue as barrier
        edge_map = np.maximum(inverted, _pp_sm)
        _pp_fine = _masked_smooth(pp_stain, tissue, 8.0)
        _pp_fine[~tissue] = 0.0
        try:
            _pp_thr = float(threshold_otsu(_pp_fine[fg])) if fg.any() else 0.5
        except Exception:
            _pp_thr = 0.5
    else:
        edge_map = inverted

    # 8-adjacency follows diagonal edges more naturally at lobule boundaries,
    # reducing staircase artefacts vs 4-adjacency.
    graph = hg.get_8_adjacency_graph((H, W))
    edge_weights = hg.weight_graph(graph, edge_map, hg.WeightFunction.max)
    labels = hg.labelisation_seeded_watershed(graph, edge_weights, markers)

    # ── Valley merge: join over-segmented pairs with no dark boundary ─────
    # Use a separate, small-sigma PV smooth (8 px fixed) so that real dark
    # valleys are NOT blurred away at large valley_sigma values.
    #
    # Protected labels (CV-seeded) are normally never merged.  Exception: when
    # PP is also dark on the shared boundary, both channels agree there is no
    # portal tract - the pair is a fused super-lobule and should be merged.
    _protected: frozenset = frozenset()
    if cv_mask is not None and cv_mask.any():
        _cv_dil1 = cv2.dilate(
            cv_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        _protected = frozenset(
            int(l) for l in np.unique(labels[_cv_dil1 & fg]) if l > 0
        )

    if fg.any():
        _smooth_merge = _masked_smooth(pv_stain, tissue, 8.0)
        try:
            # Percentile threshold is more robust than Otsu under residual
            # shading: it always selects the darkest N% of foreground pixels
            # as valleys regardless of global intensity level.
            if valley_merge_thr_pct is not None:
                valley_thr = float(np.percentile(_smooth_merge[fg], valley_merge_thr_pct))
            else:
                valley_thr = float(threshold_otsu(_smooth_merge[fg]))
            labels = _merge_below_valley(
                labels, _smooth_merge, fg, valley_thr,
                protected_labels=_protected,
                pp_smooth=_pp_fine,
                pp_thr=_pp_thr,
            )
        except Exception:
            pass  # skip merge for pathological / uniform images

    labels = _postprocess_labels(labels, fg, min_area=min_area_px)

    # ── Peak validation: merge peakless lobules ───────────────────────────────
    # Each genuine lobule has a PV brightness peak (perivenular zone near the CV).
    # Lobules without a peak are fragments - split artefacts or edge slivers.
    # They are absorbed into the neighbour with the longest shared boundary.
    labels, peak_coords = _merge_peakless_lobules(
        labels, smoothed_blob, fg, peak_thresh_pct,
    )

    # ── Fine-scale peak refinement ────────────────────────────────────────────
    # _merge_peakless_lobules returns the peak at blob_sigma scale (≈ lobule
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

    # ── Second-pass watershed from confirmed PV peaks ─────────────────────────
    # After the merge the seed set has changed.  Re-seed from the refined PV
    # brightness maxima (inside fg tissue) rather than from the original CV
    # centroids.  The same PV+PP edge-weight graph is reused - no extra
    # computation.  Every fg pixel is freshly assigned to its nearest confirmed
    # lobule centre through the strongest available boundary landscape.
    if peak_coords:
        _fresh = np.zeros((H, W), np.int32)
        _r = max(5, peak_min_dist // 8)
        for _lid, (_py, _px) in peak_coords.items():
            cv2.circle(_fresh, (int(_px), int(_py)), _r, int(_lid), -1)
        labels = hg.labelisation_seeded_watershed(graph, edge_weights, _fresh)
        labels[~fg] = 0
        labels = labels.astype(np.int32)

    # ── CV-aware spurious-fragment removal ───────────────────────────────────
    # After the second-pass watershed, small regions that contain no CV pixel
    # are likely boundary artefacts or tissue-edge slivers.  Absorb them into
    # the nearest surviving neighbour (Voronoi fill).  Regions that do contain
    # a CV are kept unconditionally - removing them would discard a genuine
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

    if boundary_smooth_sigma > 0:
        labels = _smooth_labels_valley_aware(
            labels, inverted, fg,
            sigma=boundary_smooth_sigma,
            valley_pct=boundary_smooth_valley_pct,
            valley_band_px=boundary_smooth_valley_band_px,
            pp_smooth=_pp_fine,
        )
    return labels
