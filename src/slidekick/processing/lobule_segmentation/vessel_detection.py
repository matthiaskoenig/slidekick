"""Vessel detection, classification and stain preprocessing.

This module is the single implementation of the vessel and preprocessing stage,
used by :class:`LobuleSegmentor` and by any external caller that needs the same
operations on raw arrays, the same arrangement ``segment_lobules`` already has.

Keeping vessel classification as one shared implementation is deliberate: it
used to exist twice, as two independent algorithms, and fixes landed on one copy
without the other, so the two drifted out of numeric agreement. Sharing one
implementation removes that failure mode.

Pipeline order
--------------
1. :func:`preprocess_stack`            flatfield and CLAHE, per channel
2. :func:`detect_and_classify_vessels` hole topology to CV / portal masks
3. :func:`filter_cv_seeds`             drop CV components too small to seed from
4. :func:`stains_from_stack`           capped median blur and p99 normalisation

Steps 1 and 4 both read the same corrected stack produced by step 1; step 4 is
not applied to the input of step 2. The two blurs are deliberately distinct, see
:func:`stains_from_stack`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from scipy.ndimage import gaussian_filter as _gaussian_filter
from scipy.ndimage import label as _ndlabel

from slidekick.processing.lobule_segmentation.lob_utils import holes_from_fg_mask

__all__ = [
    "apply_clahe",
    "apply_flatfield",
    "preprocess_stack",
    "stains_from_stack",
    "detect_and_classify_vessels",
    "filter_cv_seeds",
]


# ---------------------------------------------------------------------------
# Flatfield / pseudo-illumination correction
# ---------------------------------------------------------------------------

def apply_clahe(ch: np.ndarray, clip_limit: float, tile_px: int) -> np.ndarray:
    """Apply CLAHE to a float32 channel.

    Converts to uint16 for cv2.createCLAHE (preserves precision), then
    converts back to float32 scaled to the original peak value.

    *tile_px* is the desired tile size in pixels.  OpenCV's tileGridSize
    expects the number of tiles per dimension, so we convert here.
    """
    _fg = ch > 0
    if not _fg.any():
        return ch
    if clip_limit <= 0:
        return ch
    _orig_max = float(ch[_fg].max())
    _u16 = (ch / max(_orig_max, 1e-6) * 65535.0).clip(0, 65535).astype(np.uint16)
    _dim = max(ch.shape[:2])
    _grid = max(1, min(_dim, _dim // max(tile_px, 1)))
    _clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(_grid, _grid))
    _eq = _clahe.apply(_u16)
    return (_eq.astype(np.float32) / 65535.0 * _orig_max).astype(np.float32)


def apply_flatfield(ch: np.ndarray, sigma: float,
                    mask: np.ndarray | None = None,
                    mask_blur: np.ndarray | None = None) -> np.ndarray:
    """Divide channel by its masked-Gaussian local-background estimate.

    Masked-Gaussian division at *sigma*, then renormalise to the original peak
    within tissue. There is no strength or blend knob; correction is always full.

    *sigma* must be comparable to, or somewhat below, the illumination and
    shading scale, roughly one lobule width. Values several times larger
    under-correct: the local background becomes near-constant and the shading
    gradient survives.

    mask : optional boolean foreground mask (e.g. tissue mask).  When provided it
           is used instead of ``ch > 0``, which is unreliable in dark shading
           regions where tissue pixels read near-zero.  Always pass an explicit
           tissue mask when one is available; ``ch > 0`` is the fallback only.
    mask_blur : optional precomputed ``gaussian_filter(mask, sigma)``. The
           denominator blur depends only on *mask* and *sigma*, not on the
           channel, so when the same mask is reused across a stack it can be
           computed once and passed in to skip the repeat. Ignored (recomputed)
           when *mask* is None.
    """
    _ch_f = ch.astype(np.float32)
    _fg = mask.astype(bool) if mask is not None else (_ch_f > 0)
    _num = _gaussian_filter(_ch_f, sigma=sigma)
    if mask_blur is not None and mask is not None:
        _den = mask_blur
    else:
        _den = _gaussian_filter(_fg.astype(np.float32), sigma=sigma)
    _local_bg = np.where(_den > 1e-6, _num / np.maximum(_den, 1e-6), 0.0)
    _corrected = np.where(_fg, _ch_f / np.maximum(_local_bg, 1e-6), 0.0)
    _t_max = float(_corrected[_fg].max()) if _fg.any() else 0.0
    if _t_max <= 0:
        return _ch_f
    _orig_max = float(_ch_f[_fg].max())
    return (_corrected * (_orig_max / _t_max)).astype(np.float32)


def preprocess_stack(
    image_stack: np.ndarray,
    local_bg_sigma: float,
    clahe_clip_limit: float,
    clahe_tile_px: int,
    clahe_pp_factor: float = 1.0,
    has_pp: bool = False,
    tissue_mask: np.ndarray | None = None,
    pp_index: int = 1,
) -> np.ndarray:
    """Flatfield + CLAHE every channel of *image_stack*.

    Returns a float32 stack in the input's value range (nominally 0-255). Feed
    the result to both :func:`detect_and_classify_vessels` (cast to uint8) and
    :func:`stains_from_stack`; they consume the same corrected stack.

    The PP channel (only when *has_pp*) gets its clip limit scaled by
    *clahe_pp_factor*: PP is weakly zonated and needs harder equalisation than
    PV to bring the portal-tract rims out. *pp_index* says which channel that is;
    the synthetic benchmark always uses 1, but production stains can sit at any
    index.

    *tissue_mask* is passed through to :func:`apply_flatfield`. When omitted it
    is derived as ``image_stack.max(axis=-1) > 0``.
    """
    _stack_f = image_stack.astype(np.float32)
    _stack_corr = _stack_f.copy()
    _n_ch = _stack_f.shape[-1]
    if tissue_mask is None:
        tissue_mask = image_stack.max(axis=-1) > 0

    # The flatfield denominator is gaussian_filter(tissue_mask, sigma); it does
    # not depend on the channel, so blur the mask once and reuse it for every
    # channel instead of repeating the large Gaussian per channel.
    _mask_blur = (
        _gaussian_filter(tissue_mask.astype(bool).astype(np.float32),
                         sigma=local_bg_sigma)
        if local_bg_sigma > 0 else None
    )

    def _correct_channel(_c: int) -> np.ndarray:
        _ch = apply_flatfield(_stack_f[..., _c], local_bg_sigma,
                              mask=tissue_mask, mask_blur=_mask_blur)
        _clip = clahe_clip_limit
        if _c == pp_index and has_pp:
            _clip = clahe_clip_limit * clahe_pp_factor
        return apply_clahe(_ch, _clip, clahe_tile_px)

    # Each channel's flatfield + CLAHE is independent. The flatfield's large
    # Gaussian dominates runtime and scipy/cv2 release the GIL, so running the
    # channels on a thread pool gives a near-linear, bit-identical speedup (same
    # calls, each channel writes its own slice).
    if _n_ch > 1:
        with ThreadPoolExecutor(max_workers=min(_n_ch, os.cpu_count() or 1)) as _ex:
            for _c, _res in enumerate(_ex.map(_correct_channel, range(_n_ch))):
                _stack_corr[..., _c] = _res
    else:
        _stack_corr[..., 0] = _correct_channel(0)
    return _stack_corr


def stains_from_stack(
    stack_corr: np.ndarray,
    ksize: int,
    has_pp: bool = False,
    pv_index: int = 0,
    pp_index: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Derive the (pv_stain, pp_stain) pair fed to ``segment_lobules``.

    Takes the flatfield and CLAHE corrected stack from :func:`preprocess_stack`
    and applies a capped median blur, a fixed 3 px median blur, and p99
    normalisation per channel. Returns float32 arrays; ``pp_stain`` is None when
    *has_pp* is False.

    The blur is capped at 15 px even when *ksize* is larger. This is deliberate:
    PV peaks sit next to central veins only 16-48 px across, and an unclamped
    25 px kernel blurs them away. Keep this distinct from the detector's own
    ``vessel_blur_px``; the two blurs serve different stages and must not be
    collapsed into one knob.
    """
    _img = np.clip(stack_corr, 0, 255).astype(np.uint8)
    for _c in range(_img.shape[-1]):
        _ch = _img[..., _c]
        _k = int(ksize)
        if _k % 2 == 0:
            _k += 1
        _k = min(_k, 15)
        if _k >= 3:
            _ch = cv2.medianBlur(_ch, _k)
        _ch = cv2.medianBlur(_ch, 3)
        _nz = _ch[_ch > 0]
        _p99 = float(np.percentile(_nz, 99)) if _nz.size else 255.0
        if _p99 <= 0:
            _p99 = 1.0
        _img[..., _c] = (
            (_ch.astype(np.float32) / _p99 * 255.0).clip(0, 255).astype(np.uint8)
        )

    pv_stain = _img[:, :, pv_index].astype(np.float32)
    pp_stain = _img[:, :, pp_index].astype(np.float32) if has_pp else None
    return pv_stain, pp_stain


# ---------------------------------------------------------------------------
# Vessel detection pipeline
# ---------------------------------------------------------------------------

def detect_and_classify_vessels(
    image_stack_u8: np.ndarray,
    has_pp: bool,
    ksize: int,
    vessel_annulus_px: int,
    vessel_circularity_min: float,
    vessel_max_area_px: int,
    vessel_blur_px: int,
    min_vessel_area_pp: int,
    min_vessel_area_pv: int,
    vessel_tophat_frac: float,
    vessel_tophat_kernel_px: int,
    vessel_pv_zone_thr: float = 0.35,
    vessel_pp_zone_thr: float = 0.9,
    vessel_zone_sigma: float = 20.0,
    vessel_min_enclosure: float = 0.9,
    pv_index: int = 0,
    pp_index: int = 1,
    stain_indices=None,
    vessel_lumen_pct=None,
    vessel_lumen_blur_px: int = 5,
    **_ignored,
) -> tuple:
    """Zone-based vessel classification.

    *vessel_lumen_pct* selects how a lumen is recognised: None keeps the
    exact-zero test (correct for synthetic and pre-zeroed images), a percentile
    switches to a median-blurred relative threshold (correct for real scans). See
    the comment at the tissue_mask computation for the trade-off.

    *stain_indices* names the channels used for the lumen projection, the ones
    selected as PV or PP. Tissue extent still comes from every channel; only the
    hole and lumen test is restricted, because a structural channel (nuclear,
    membrane) is bright inside lumens and would fill them. None falls back to all
    channels, which is correct only when no structural channel is present (for
    example the synthetic generator); real callers should pass their stain
    channels.

    Detects vessel holes via foreground topology, then classifies each hole as CV
    or portal by the zone composition of an annular ring. With PV only,
    ring_pv > fg_mean marks a CV. With PV and PP, it builds PV/PP zone masks from
    the smoothed stains and checks what fraction of the ring falls in each zone.

    *image_stack_u8* is the corrected stack from :func:`preprocess_stack`, cast
    to uint8, not the raw image and not the output of :func:`stains_from_stack`.

    Known limitation: hole-topology detection can miss real central veins
    entirely. A lumen only becomes a hole (via ``holes_from_fg_mask``) if it reads
    as background in every channel contributing to the tissue mask. On a real
    multiplexed slide, structural channels (nuclear, membrane) commonly carry
    signal inside the lumen, so it never registers as a hole no matter how the
    thresholds are set, and this is not tunable away. Seeding is unaffected
    because ``_find_intensity_cluster_seeds`` carries it through the intensity-peak
    fallback when hole detection finds nothing; that fallback is load-bearing, not
    a backup. If CV detection matters beyond seeding (counting or measuring central
    veins), hole topology is not sufficient here and would need a detector based on
    intensity minima.

    *vessel_max_area_px* is a fixed multiple of the calibration lobule (scaled by
    ``LobuleSegmentor.scaled_defaults`` along with the other pixel-valued
    parameters), not a measurement of real vessel-size variability. At correct
    scale the largest real hole candidates sit below the shipped ceiling, so it is
    not what limits detection, but a fibrotic or dilated vessel several times the
    typical size could still exceed it. Raise the ceiling if you expect unusually
    large vessels.

    Returns (cv_mask, pf_mask, tissue_mask, vessel_holes), all bool (H, W).
    """
    H, W = image_stack_u8.shape[:2]

    # Optional median blur (preserves small CV holes when 0)
    k_vessel = int(vessel_blur_px)
    if k_vessel % 2 == 0 and k_vessel > 0:
        k_vessel += 1
    if k_vessel >= 3:
        filtered = np.stack(
            [cv2.medianBlur(image_stack_u8[:, :, c], k_vessel)
             for c in range(image_stack_u8.shape[2])],
            axis=-1,
        )
    else:
        filtered = image_stack_u8

    # Tissue mask from a max-projection threshold.
    #
    # A bare ``> 0`` threshold leaves a ragged border, which causes two distinct
    # defects. (1) Near-enclosed concavities in the outline survive the
    # largest-contour fill inside holes_from_fg_mask and get detected as vessels,
    # producing CV seeds on the tissue edge. (2) Small disconnected foreground
    # islands along the border never receive a watershed label, and the Voronoi
    # fill in _postprocess_labels then paints each one with the nearest distant
    # label, the stray specks in the overlay. An open-then-close plus
    # largest-component selection removes both at source.
    #
    # Two projections with different jobs. Tissue extent uses every channel: a
    # structural channel (nuclear, membrane) legitimately marks where tissue is,
    # and a periportal region can be dark in a pericentral stain, so deriving
    # extent from the stains alone drops real tissue.
    _proj_tissue = filtered.max(axis=2)
    #
    # Lumen and hole evidence uses the stain channels only. A structural channel
    # is bright throughout the tissue, inside vessel lumens included, so letting it
    # into the hole projection fills every lumen and the enclosed-hole test finds
    # nothing. On a slide whose channel 0 is a DAPI tissue marker, the all-channel
    # projection produces no hole candidates while stain-only recovers them.
    _lumen = ([c for c in dict.fromkeys(stain_indices)
               if 0 <= c < filtered.shape[2]] if stain_indices else [])
    if not _lumen:
        _lumen = list(range(filtered.shape[2]))
    _proj_lumen = filtered[:, :, _lumen].max(axis=2)

    # How a lumen is recognised in the stain projection. Max is the right
    # projection: PV and PP are reciprocal, each bright near the opposite vessel
    # type, so tissue is bright in at least one stain while a lumen is bright in
    # neither. What differs between image classes is the rule:
    #
    #   None (default), ``> 0``. Correct wherever lumens are encoded as exact
    #     zeros (the synthetic generator, and any pre-zeroed background).
    #
    #   a percentile, median-blur then threshold at that percentile of the tissue
    #     interior. Real scans never reach exact zero inside tissue, so ``> 0``
    #     finds almost nothing there; the percentile recovers the lumens.
    #
    # These are not interchangeable. Applying the percentile rule to synthetic
    # data collapses precision, because synthetic tissue legitimately approaches
    # zero in low-zonation regions and a relative threshold carves it into false
    # holes. Auto-detecting the regime is unreliable: the exact-zero fraction
    # barely separates the two image classes.
    tissue_mask = _proj_tissue > 0
    if vessel_lumen_pct is None:
        _stain_bright = _proj_lumen > 0
    else:
        _k = int(vessel_lumen_blur_px)
        if _k % 2 == 0:
            _k += 1
        _sm = cv2.medianBlur(_proj_lumen, _k) if _k >= 3 else _proj_lumen
        _int = _sm[tissue_mask & (_sm > 0)]
        _thr = float(np.percentile(_int, float(vessel_lumen_pct))) if _int.size else 0.0
        _stain_bright = _sm > _thr

    # Clean only the outer contour. Applying open/close to the mask directly
    # erases lumens: MORPH_CLOSE fills small holes, and vessel lumens are small
    # holes. Smoothing the hole-filled silhouette and then re-subtracting the
    # original interior holes gives a clean border while leaving every lumen
    # untouched.
    _interior = binary_fill_holes(tissue_mask) & ~tissue_mask
    _solid = binary_fill_holes(tissue_mask).astype(np.uint8)
    _tk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _solid = cv2.morphologyEx(_solid, cv2.MORPH_OPEN, _tk)
    _solid = cv2.morphologyEx(_solid, cv2.MORPH_CLOSE, _tk)
    _n_cc, _cc_lab, _cc_stats, _ = cv2.connectedComponentsWithStats(_solid, 8)
    if _n_cc > 1:
        _largest = 1 + int(np.argmax(_cc_stats[1:, cv2.CC_STAT_AREA]))
        _solid_b = _cc_lab == _largest
    else:
        _solid_b = _solid.astype(bool)
    tissue_mask = _solid_b & ~_interior

    # Hole-topology vessel detection on the stain projection. Foreground for the
    # topology is tissue that is stain-bright; a stain-dark region enclosed by it
    # is a lumen and becomes a hole. Deriving holes from tissue_mask directly
    # would miss every lumen a structural channel keeps bright (the DAPI case).
    _fg_holes = tissue_mask & _stain_bright
    # No ``& tissue_mask`` here: a lumen that reads zero in every channel (the
    # synthetic generator, any pre-zeroed background) is not part of tissue_mask,
    # so intersecting would delete exactly those lumens. holes_from_fg_mask
    # already returns only the interior of the foreground silhouette, so the holes
    # are enclosed by tissue by construction.
    vessel_holes = holes_from_fg_mask(_fg_holes)

    # Optional top-hat punch for missed lumens
    if vessel_tophat_frac > 0:
        gray = _proj_lumen
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (vessel_tophat_kernel_px, vessel_tophat_kernel_px),
        )
        bhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern)
        thr = vessel_tophat_frac * 255.0
        tophat_holes = (bhat > thr) & tissue_mask
        vessel_holes = vessel_holes | tophat_holes

    # Split into connected components, filter by area + circularity
    n_cc, labels_cc = cv2.connectedComponents(vessel_holes.astype(np.uint8), connectivity=8)

    pv = filtered[:, :, pv_index].astype(np.float32)
    pp = filtered[:, :, pp_index].astype(np.float32) if has_pp else None

    fg_tissue = tissue_mask & ~vessel_holes

    # Precompute global stats
    fg_pv_mean = float(pv[fg_tissue].mean()) if fg_tissue.any() else 0.5
    # CV-stain composite for the no-PP classification fallback below. _proj_lumen
    # is the max over the selected stain channels; with no PP channel that is the
    # pericentral (PV) marker set, so a central vein (bright pericentral cytoplasm
    # around a dark lumen) reads high here, and using the max of several PV markers
    # is more robust than any single one.
    fg_cvstain_mean = float(_proj_lumen[fg_tissue].mean()) if fg_tissue.any() else 0.5

    # Build pixel-level zone masks
    pv_zone = None
    pp_zone = None
    if pp is not None and vessel_zone_sigma > 0:
        pv_smooth = _gaussian_filter(pv, sigma=vessel_zone_sigma)
        pp_smooth = _gaussian_filter(pp, sigma=vessel_zone_sigma)
        pv_median = float(np.median(pv_smooth[fg_tissue])) if fg_tissue.any() else 0.5
        pp_median = float(np.median(pp_smooth[fg_tissue])) if fg_tissue.any() else 0.5
        pv_zone = (pv_smooth > pv_median) & fg_tissue
        pp_zone = (pp_smooth > pp_median) & fg_tissue

    cv_mask = np.zeros((H, W), dtype=bool)
    pf_mask = np.zeros((H, W), dtype=bool)

    for cc_id in range(1, n_cc):
        hole = labels_cc == cc_id
        area = int(hole.sum())
        # Gate on the more permissive of the two class thresholds; the
        # class-specific one is applied after classification. min_vessel_area_pv
        # is the CV path's own area floor.
        min_area = min(min_vessel_area_pp, min_vessel_area_pv)
        if area < min_area or area > vessel_max_area_px:
            continue

        contours, _ = cv2.findContours(
            hole.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter ** 2)
        if circ < vessel_circularity_min:
            continue

        # Enclosure test. A real lumen is surrounded by tissue on every side; a
        # concavity in the outer tissue outline is not, but can still register as
        # a hole because holes_from_fg_mask fills the largest external contour and
        # closes over the notch. Such border artefacts otherwise become CV seeds
        # on the tissue edge. Comparing the full dilation ring against the part of
        # it that lies inside tissue separates the two cleanly and reuses the
        # annulus already computed.
        _ring_all = binary_dilation(hole, iterations=vessel_annulus_px) & ~hole
        ring = _ring_all & fg_tissue
        _ring_all_n = int(_ring_all.sum())
        if _ring_all_n > 0:
            _enclosure = float(ring.sum()) / _ring_all_n
            if _enclosure < vessel_min_enclosure:
                continue
        if ring.sum() < 10:
            continue

        if pv_zone is not None and pp_zone is not None:
            ring_n = int(ring.sum())
            pv_frac = float(np.count_nonzero(ring & pv_zone)) / ring_n
            pp_frac = float(np.count_nonzero(ring & pp_zone)) / ring_n
            pv_pass = pv_frac >= vessel_pv_zone_thr
            pp_pass = pp_frac >= vessel_pp_zone_thr

            # PP-priority tie-break.
            #
            # pv_zone is a median split, so pv_frac sits near 0.5 by construction
            # and pv_pass is trivially true for any sensible vessel_pv_zone_thr
            # (0.2-0.5). Sending every pv_pass candidate to cv_mask would leave
            # pf_mask structurally unreachable. Deciding the both-pass case by
            # which fraction is larger, with PP taking the tie, keeps portals
            # reachable.
            if pp_pass and (not pv_pass or pp_frac >= pv_frac):
                if area >= min_vessel_area_pp:
                    pf_mask |= hole
            elif pv_pass:
                if area >= min_vessel_area_pv:
                    cv_mask |= hole
            else:
                mean_pv = float(pv[ring].mean())
                if mean_pv > fg_pv_mean:
                    if area >= min_vessel_area_pv:
                        cv_mask |= hole
                elif area >= min_vessel_area_pp:
                    pf_mask |= hole
        else:
            # No PP channel: classify by pericentral-stain brightness in a ring
            # offset from the lumen edge. The immediate perivascular cuff
            # (~annulus px) is a dark space that reads near zero for every vessel;
            # the pericentral cytoplasm that marks a central vein sits a band
            # further out. On a weakly-zonated slide the immediate annulus reads
            # zero for both vessel types, while the annulus to 5*annulus band
            # separates a central vein's bright pericentral cytoplasm from a
            # portal's dim surround.
            _r_out = binary_dilation(hole, iterations=int(vessel_annulus_px) * 5)
            _r_in = binary_dilation(hole, iterations=int(vessel_annulus_px))
            class_ring = (_r_out & ~_r_in) & fg_tissue
            mean_cv = (float(_proj_lumen[class_ring].mean())
                       if class_ring.any() else 0.0)
            if mean_cv > fg_cvstain_mean:
                if area >= min_vessel_area_pv:
                    cv_mask |= hole
            elif area >= min_vessel_area_pp:
                pf_mask |= hole

    if vessel_lumen_pct is not None:
        # Under the percentile rule the raw hole set is mostly noise (most
        # components are under 50 px), and ``segment_lobules`` subtracts
        # vessel_holes from its foreground (``fg = tissue & ~holes``). Passing the
        # raw set would remove a large fraction of the tissue, including real
        # lobule area, from the region the watershed may segment. Only holes that
        # survived classification are real lumens, so only those are reported.
        # tissue_mask is already the full-extent mask (extent is decoupled from
        # the lumen projection), so it needs no reversion here.
        vessel_holes = (cv_mask | pf_mask) & tissue_mask

    return cv_mask, pf_mask, tissue_mask, vessel_holes


def filter_cv_seeds(cv_mask: np.ndarray, min_cv_seed_area: int) -> np.ndarray:
    """Drop CV components below *min_cv_seed_area* from a seed mask.

    Decouples scoring from seeding: the caller keeps the full ``cv_mask`` for
    detection metrics and portality, and passes the filtered copy to
    ``segment_lobules`` so small noisy components do not become watershed seeds.
    Returns a new array; *cv_mask* is not modified.

    A modest floor (around 100 px) only rejects tiny duplicate components on
    lobules that already carry a valid CV; it starts costing whole lobules a seed
    only at much larger values.
    """
    out = cv_mask.copy()
    if min_cv_seed_area <= 0:
        return out
    _lbl_arr, _n_cc = _ndlabel(out)
    for _cid in range(1, _n_cc + 1):
        _comp = _lbl_arr == _cid
        if int(_comp.sum()) < min_cv_seed_area:
            out[_comp] = False
    return out
