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
    peaks = peaks & (
        ((blurred > thr_interior) & ~edge_zone) |
        ((blurred > thr_edge) & edge_zone)
    )

    # Step 3: NMS — sort by brightness, suppress nearby duplicates
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return []

    brightness = blurred[ys, xs]
    order = np.argsort(-brightness)
    ys, xs, brightness = ys[order], xs[order], brightness[order]

    kept_peaks: list[tuple[int, int]] = []
    for y, x, b in zip(ys, xs, brightness):
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


def _smooth_labels_valley_aware(
    labels: np.ndarray,
    inverted: np.ndarray,
    fg: np.ndarray,
    sigma: float = 5.0,
    valley_pct: float = 75.0,
    valley_band_px: int = 2,
) -> np.ndarray:
    """Smooth label boundaries while preserving intensity valleys.

    Each label's binary mask is blurred with a Gaussian of radius *sigma*
    pixels. Every fg pixel is then (re-)assigned to the label with the
    highest blurred score at that location — effectively rounding the
    boundary curves.

    Pixels that sit on a real intensity valley are **frozen**: they keep
    their original label and are excluded from reassignment. A narrow dilation
    band around each valley is also frozen so the smoothed boundary cannot
    erode into the valley.

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
        Gaussian smoothing radius in pixels. Larger → rounder boundaries.
        0 disables the step entirely.
    valley_pct : float
        Percentile of the fg *inverted* distribution above which a pixel is
        treated as a real valley and frozen. E.g. 75 freezes the darkest
        25 % of the fg area. Lower → more pixels frozen (less smoothing but
        valley boundaries faithfully preserved); higher → fewer frozen (more
        smoothing, accepts some valley distortion).
    valley_band_px : int
        Dilation radius (px) applied around frozen valley pixels. Prevents
        the smoothed boundary from creeping up to or across the valley.

    Returns
    -------
    np.ndarray
        Smoothed int32 label map. Background and vessel-hole pixels remain 0.
    """
    if sigma <= 0.0:
        return labels.astype(np.int32, copy=True)

    from scipy.ndimage import binary_dilation

    fg_vals = inverted[fg]
    if len(fg_vals) == 0:
        return labels.astype(np.int32, copy=True)

    valley_thr = float(np.percentile(fg_vals, valley_pct))

    # Build frozen mask: real valleys + a surrounding buffer band.
    valley_core = fg & (inverted > valley_thr)
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
    """Zero labels outside *fg* and remove regions smaller than *min_area* pixels.

    Parameters
    ----------
    labels:
        Integer label map from watershed.
    fg:
        Boolean foreground mask.
    min_area:
        Minimum region size in pixels; smaller regions are zeroed out.

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
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_lobules(
    tissue: np.ndarray,
    vessel_holes: np.ndarray,
    pv_stain: np.ndarray,
    *,
    blob_sigma: float = 52.0,
    peak_min_dist: int = 120,
    peak_thresh_pct: float = 52.0,
    valley_sigma: float = 12.0,
    cluster_merge_sigma: float = 0.0,
    min_area_px: int = 5000,
    boundary_smooth_sigma: float = 20.0,
    boundary_smooth_valley_pct: float = 90.0,
    boundary_smooth_valley_band_px: int = 2,
) -> np.ndarray:
    """Segment liver lobules via Higra seeded watershed.

    Seeds are detected as local intensity maxima in the masked-Gaussian-smoothed
    PV-channel image. The watershed runs on an inverted smoothed field so that
    dark valleys (inter-lobule boundaries) correspond to high edge weights.

    Parameters
    ----------
    tissue:
        Boolean mask of foreground tissue (full image extent).
    vessel_holes:
        Boolean mask of vessel interiors to exclude from seeding.
    pv_stain:
        Float32 PV-channel intensity image (same spatial extent as *tissue*).
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

    seed_cnts = _find_intensity_cluster_seeds(
        pv_stain,
        fg,
        np.zeros((H, W), dtype=bool),
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

    graph = hg.get_4_adjacency_graph((H, W))
    edge_weights = hg.weight_graph(graph, inverted, hg.WeightFunction.max)
    labels = hg.labelisation_seeded_watershed(graph, edge_weights, markers)

    labels = _postprocess_labels(labels, fg, min_area=min_area_px)
    if boundary_smooth_sigma > 0:
        labels = _smooth_labels_valley_aware(
            labels, inverted, fg,
            sigma=boundary_smooth_sigma,
            valley_pct=boundary_smooth_valley_pct,
            valley_band_px=boundary_smooth_valley_band_px,
        )
    return labels
