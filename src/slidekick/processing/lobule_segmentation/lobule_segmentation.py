import numpy as np
from pathlib import Path
from rich.prompt import Confirm
from typing import List, Union, Optional, Tuple, Dict, Any
import cv2
from scipy.ndimage import gaussian_filter as _gaussian_filter
import datetime
import inspect
import uuid

import matplotlib.pyplot as plt

from slidekick.processing.baseoperator import BaseOperator
from slidekick.io.metadata import Metadata
from slidekick.console import console
from slidekick import OUTPUT_PATH
from slidekick.io import save_tif

from slidekick.processing.roi.roi_utils import largest_bbox, ensure_grayscale_uint8, crop_image, detect_tissue_mask
from slidekick.processing.lobule_segmentation.portality import (
    LUMEN_DISPLAY_RGB,
    lobule_portality,
    vessel_lumen_mask,
)
from slidekick.processing.lobule_segmentation.vessel_detection import (
    detect_and_classify_vessels,
    filter_cv_seeds,
    preprocess_stack,
)
from slidekick.processing.lobule_segmentation.lob_utils import (
    detect_tissue_mask_multiotsu, overlay_mask,
    downsample_stack_to_max_side, minmax_to_uint8,
    to_base_full,
    common_pyramid_levels, choose_default_preview_level, load_level_from_multiscale,
    discover_pyramid_shapes,
)
from slidekick.processing.lobule_segmentation.napari_utils import (
    preview_images_napari,
    run_napari_preview_action,
)

# ---------------------------------------------------------------------------
# Visualisation colour palette (BGR for OpenCV)
# ---------------------------------------------------------------------------
_COLOR_PP: tuple = (255, 0, 255)    # magenta, periportal zone
_COLOR_MID: tuple = (60, 160, 60)   # green, mid zone
_COLOR_PV: tuple = (0, 165, 255)    # orange, perivenous zone


class PipelineBack(Exception):
    """
    Raised when the user requests to move backwards in the interactive napari pipeline.

    `target_step` is interpreted by `apply()`'s simple wizard state machine.
      0 = preprocessing preview
      1 = load + filter
      2 = weighting preview
      3 = vessel detection
    """

    def __init__(self, target_step: int, message: str = "Back requested"):
        super().__init__(message)
        self.target_step = int(target_step)


class PipelineAbort(Exception):
    """
    Raised when the user presses ABORT in any interactive napari preview.

    This is handled at the top-level `apply()` to exit the pipeline cleanly without
    producing new segmentations.
    """

    def __init__(self, message: str = "Abort requested"):
        super().__init__(message)


def _diag_label_for_preview(mask):
    """Component count for a boolean mask (preview diagnostics only)."""
    from scipy.ndimage import label as _lbl
    lab, n = _lbl(mask)
    return n, lab


def _masks_to_contours(cv_mask: np.ndarray,
                       pf_mask: np.ndarray) -> Tuple[List[int], list]:
    """Adapt boolean CV/portal masks to the (classes, contours) pair.

    ``vessel_detection.detect_and_classify_vessels`` returns masks, while the
    rest of ``apply()`` (debug overlays, the napari previews and the portality
    rasterisation) consumes OpenCV contours tagged with a class (0 = central
    vein, 1 = portal). Converting here keeps those consumers untouched instead of
    rewiring each one.
    """
    classes: List[int] = []
    contours: list = []
    for _cls, _mask in ((0, cv_mask), (1, pf_mask)):
        if _mask is None or not _mask.any():
            continue
        _cnts, _ = cv2.findContours(_mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for _c in _cnts:
            contours.append(_c)
            classes.append(_cls)
    return classes, contours


# ---------------------------------------------------------------------------
# Scale reference for the default parameters
# ---------------------------------------------------------------------------
# Every pixel-valued default below is tied to one scale.  Measured on the 60
# human-annotated lobule faces of the reference slide, the median face has an
# equivalent diameter (2*sqrt(area/pi)) of 226.4 px at the level the defaults
# were tuned on.  That number is a MEASUREMENT, and it is the anchor.
#
# Running at a different magnification without rescaling is not a small effect:
# at 2x the reference scale the defaults lose 0.070 mIoU (4.0x the metric's own
# standard error) and over-segment 2.6-fold, because the area gates then admit
# regions a quarter of the intended size.  Rescaling recovers that to within noise.
CALIBRATION_LOBULE_DIAMETER_PX = 226.4
# Pyramid level the measurement above was made at. Used only to warn when a
# caller is at a different level with no scale argument. It is not proof the
# level is wrong (a different slide's level 3 may not match this scale at all),
# only the one case this can actually detect: nothing was set, and the level
# is not the one used to derive CALIBRATION_LOBULE_DIAMETER_PX.
CALIBRATION_BASE_LEVEL = 3

# Used only to turn a micrometres-per-pixel figure into the pixel anchor above.
# Unlike CALIBRATION_LOBULE_DIAMETER_PX this is an assumption, not a measurement:
# a classic human hepatic lobule is roughly 1 mm across. Sanity check: it places
# the reference slide at 0.55 um/px at full resolution, the same magnification
# class as a co-located slide that reports 0.44 um/px in its own metadata.
TYPICAL_LOBULE_DIAMETER_UM = 1000.0

# Parameters that scale with magnification. Lengths scale linearly, areas with
# the square. Everything not listed (percentiles, ratios, blends, and `pp_thr`,
# a percentile, see higra_segment) is scale-free and must not be touched.
_SCALE_LENGTH = (
    "blob_sigma", "peak_min_dist", "valley_sigma", "valley_merge_sigma",
    "boundary_smooth_sigma", "boundary_smooth_valley_band_px",
    "contour_smooth_sigma", "edge_local_contrast_sigma", "min_seed_border_dist",
    "seed_tissue_ring", "local_bg_sigma", "clahe_tile_px", "vessel_annulus_px",
    "vessel_zone_sigma", "vessel_blur_px", "ksize", "vessel_tophat_kernel_px",
)
_SCALE_AREA = (
    "min_area_px", "min_vessel_area_pp", "min_vessel_area_pv",
    "vessel_max_area_px", "min_cv_seed_area",
)
"""Areas that scale with the square of the magnification ratio.

The vessel-area gates belong here because the scale knob is a magnification
ratio: at a coarser or finer pyramid level every structure in the image changes
size together (lobules, vessel lumens, nuclei, noise grain), so a vessel gate
must follow. Leaving them out only makes sense for a specimen whose lobules
differ in size at the same magnification, which the scale knob cannot express
(see :meth:`LobuleSegmentor.scale_factor`); without them ``vessel_max_area_px``
stays at its calibration value for a coarser-level caller whose vessels are many
times that area, rejecting every large vessel.
"""


class LobuleSegmentor(BaseOperator):
    """
    Based on https://github.com/matthiaskoenig/zonation-image-analysis/blob/develop/src/zia/pipeline/pipeline_components/segementation_component.py

    Where the default parameters come from
    --------------------------------------
    The vessel-detection, seeding and watershed defaults below are a tuned
    configuration, not hand-picked values, selected by a parameter search over a
    blend of synthetic and real scores. On the hand-annotated fluorescence slide
    (a 6-plex simliva section, 60 annotated lobule faces) this configuration
    reaches a quality-weighted mean IoU of about 0.717 against the human
    annotation, scoring each face by the annotator's own confidence grade. A
    default-constructed operator reproduces that number.

    What the configuration does and does not do:

    * Real-slide IoU is close to the previous shipped configuration; the gains the
      search found were mostly in the vessel and zonation terms, which have ground
      truth only on the synthetic benchmark. So this is a better vessel and
      zonation configuration there, not a measurably better lobule segmenter on
      real tissue.
    * The permissive vessel gates are inert on real fluorescence. On the real
      slide the detector still finds no central veins, because real lumens are not
      channel-wide zeros and no CV hole forms at any threshold. Real portality
      runs on the ``_find_cv_seed`` brightness surrogate.

    Caveats:

    * One slide. A single hand-annotated section, half of whose faces the
      annotator marked low quality. Treat 0.717 as a measurement on that slide,
      not as an expected accuracy on new tissue.
    * Absolute pixels, and rescaling is opt-in. Every length and area below is in
      pixels at the magnification the tuning ran at, where a typical lobule
      measures ``CALIBRATION_LOBULE_DIAMETER_PX`` across. Nothing rescales itself
      with ``base_level``. At a different magnification pass ``lobule_diameter_px``
      (or ``um_per_px``); lengths scale linearly, areas quadratically. Running
      unscaled at twice the calibration scale costs mIoU and over-segments
      badly.
    * It over-segments (around 79 predicted lobules against 60 annotated faces).
      Under the scoring used a split costs far less than a merge, so the
      configuration is biased that way on purpose.
    * Three defaults are range-limited rather than optimised: each landed on the
      edge of its search interval, so it is the best value inside a range that
      turned out too narrow. At their ceiling: ``vessel_max_area_px`` and
      ``min_seed_border_dist``. At its floor: ``vessel_min_enclosure``. A future
      tuning round should widen these first.

    Vessel detection is the pixel-level zone classifier in ``vessel_detection``,
    shared with every other caller so there is exactly one implementation of this
    stage. A second implementation once existed alongside it (superpixels and
    weighted K-Means); on the same slide the two were statistically
    indistinguishable, so it was removed in favour of the simpler path rather than
    on any performance grounds.
    """

    def __init__(self,
                 metadata: Union[Metadata, List[Metadata]],
                 channel_selection: Union[int, List[int]] = None,
                 channels_pp: Union[int, List[int]] = None,
                 channels_pv: Union[int, List[int]] = None,
                 throw_out_ratio: float = None,
                 preview: bool = True,
                 confirm: bool = True,
                 base_level: int = 0,
                 multi_otsu: Optional[bool] = None,
                 ksize: int = 5,
                 adaptive_histonorm: bool = True,
                 bg_low_val: int = 0,
                 clahe_bg_suppress: int = 0,
                 local_bg_correction: bool = True,
                 local_bg_sigma: float = 450.0,
                 clahe_clip_limit: float = 0.5,
                 clahe_tile_px: int = 224,
                 clahe_pp_factor: float = 2.5,
                 fg_min_signal_frac: float = 0.05,
                 # vessel gating
                 min_vessel_area_pp: int = 75,
                 min_vessel_area_pv: int = 75,
                 vessel_annulus_px: int = 6,
                 vessel_circularity_min: float = 0.06,
                 # 0.0, not 0.30: the black-hat punch is actively harmful. At 0.30
                 # the 50 px kernel crosses threshold on ordinary tissue texture and
                 # punches out most of the tissue as vessel holes, which are then
                 # subtracted from the watershed foreground. Nonzero values also
                 # produce catastrophic outliers on synthetic data.
                 vessel_tophat_frac: float = 0.0,
                 vessel_tophat_kernel_px: int = 45,
                 vessel_max_area_px: int = 30000,
                 vessel_blur_px: int = 5,
                 vessel_pv_zone_thr: float = 0.55,
                 vessel_pp_zone_thr: float = 0.85,
                 vessel_zone_sigma: float = 40.0,
                 vessel_min_enclosure: float = 0.6,
                 min_cv_seed_area: int = 150,
                 min_area_px: int = 4000,
                 pp_close_radius: int = 8,
                 # Higra seeded watershed parameters
                 blob_sigma: float = 15.0,
                 peak_min_dist: int = 120,
                 peak_thresh_pct: float = 45.0,
                 valley_sigma: float = 17.5,
                 cluster_merge_sigma: float = 0.0,
                 valley_merge_thr_pct: Optional[float] = 57.5,
                 pp_thr: Optional[float] = 100.0,
                 pp_blend: float = 0.8,
                 peak_thresh_factor: float = 0.0,
                 contour_smooth_sigma: float = 6.0,
                 min_seed_border_dist: float = 20.0,
                 higra_preview_max_px: int = 1500,
                 boundary_smooth_sigma: float = 25.0,
                 boundary_smooth_valley_pct: float = 35.0,
                 boundary_smooth_valley_band_px: int = 15,
                 # Edge-map / merge / seed-geometry controls.  These were previously
                 # hardcoded to segment_lobules' own defaults at the call sites, so a
                 # tuned configuration could not be reproduced here at all: two of them
                 # (edge_local_contrast_sigma, fused_merge_pv_pct) default to OFF while
                 # tuning finds them active.  Defaults below match segment_lobules, so
                 # existing behaviour is unchanged unless a caller sets them.
                 edge_grad_blend: float = 0.7,
                 edge_local_contrast_sigma: float = 20.0,
                 fused_merge_pv_pct: Optional[float] = 35.0,
                 seed_tissue_ring: int = 28,
                 valley_merge_sigma: float = 16.0,
                 # Opt-in rescaling; both None (the default) means no rescaling
                 # at all and behaviour identical to before this existed.
                 # See CALIBRATION_LOBULE_DIAMETER_PX above.
                 lobule_diameter_px: Optional[float] = None,
                 um_per_px: Optional[float] = None,
                 vessel_lumen_pct: Optional[float] = None,
                 vessel_lumen_blur_px: int = 5,
                 vessel_portality_enclosure: Optional[float] = None,
                 ):
        """
        @param metadata: List or single metadata object to load and use for lobule segmentation. All objects used should be single channel and either periportal or pericentrally expressed.
        @param channel_selection: list of channel indices to invert (an int is accepted for
            a single one); nothing is selected despite the name. Invert channels whose
            expression is brighter pericentrally, so that every channel ends up brighter
            periportally. None inverts nothing.
            Note: stored as `self.channels`, which in BaseOperator means something else
            entirely (selected (image, channel) tuples). The override is deliberate but the
            shared attribute name is a trap, so read this class, not the base, for its meaning.
        @param channels_pp: Channel[s] that are brighter in periportal regions (potentially after inversion). At least one channel has to be set for channel_pp or channel_pv. If only one is provided, the other will be calculated as the furthest distance on the same channel(s).
        @param channels_pv: Channel[s] that are brighter in periveneous regions (potentially after inversion). At least one channel has to be set for channel_pp or channel_pv. If only one is provided, the other will be calculated as the furthest distance on the same channel(s).
        @param throw_out_ratio: minimum fraction (0 to 1) of non-background pixels required for the slide to be processed; slides below this threshold are discarded
        @param preview: whether to preview the segmentation
        @param confirm: whether to confirm the segmentation
        @param base_level: Pyramid level to load
        @param multi_otsu: tissue detection mode.
            None (default) auto-detects: runs 3-class multi-Otsu and checks the fraction of
            pixels in the middle intensity class (mic_bg_frac). If mic_bg_frac < 5% the
            histogram is treated as bimodal (good export, no microscopy artefact) and both
            middle and top classes become tissue. If mic_bg_frac >= 5% the histogram is
            trimodal (microscopy background present) and only the top class is tissue.
            True forces 3-class behaviour (tissue = top class only, old default).
            False forces 2-class simple Otsu (no microscopy background handling).
        @param ksize: kernel size for convolution in filtering (cv2.median blur), must be odd and greater than 1, e.g., 3, 5, 7, ...
        @param adaptive_histonorm: use adaptive histogram norm in filtering
        @param local_bg_correction: if True, apply per-channel pseudo-flatfield correction in
            _filter() to remove slow-varying illumination gradients (uneven staining, vignetting).
            Each channel is divided by a masked Gaussian of itself at *local_bg_sigma* radius,
            making intensities more spatially uniform before classification.
        @param local_bg_sigma: Gaussian sigma (pixels, at the loaded pyramid level) for the
            background estimate used by local_bg_correction. Set it comparable to, or somewhat
            below, the illumination and shading scale, roughly one lobule width. Larger values
            (several lobule widths) under-correct: the local-background estimate becomes nearly
            constant and the shading gradient survives. Tune it live in the interactive
            preprocessing preview (watch the tissue flatten without the zonation washing out).
        @param min_vessel_area_pp: min area (px) for PP-enclosed vessel candidates
        @param min_vessel_area_pv: min area (px) for PV-enclosed vessel candidates
        @param vessel_annulus_px: thickness (px) of ring-consistency check
        @param vessel_circularity_min: 4πA/P^2 gate; low keeps elongated vessels
        @param min_area_px: define area threshold to filter out very small lobule regions (noise)
        @param pp_close_radius: radius (px) for morphological closing of the PP zone mask
            before vessel classification. 0 = disabled. A small value (e.g. 5 to 15) bridges
            narrow gaps in the PP zone.
        @param blob_sigma: Gaussian sigma for PV-peak detection in the Higra watershed seeding
            (should approximate the lobule radius in pixels).
        @param peak_min_dist: minimum pixel distance between kept peaks (NMS radius) for
            Higra watershed seeding.
        @param peak_thresh_pct: percentile brightness threshold for interior PV peaks.
        @param valley_sigma: Gaussian sigma for the valley-smoothed field used as Higra
            watershed edge weights. Larger values produce smoother lobule boundaries.
        @param cluster_merge_sigma: coarse-scale sigma for basin-merging in seed detection;
            0 or None disables merging. "cluster" here means a cluster of intensity-peak
            seeds; it predates, and is unrelated to, the K-Means clustering that this
            module no longer contains.
        @param valley_merge_thr_pct: percentile of the smoothed-PV distribution used as the
            valley-merge threshold (adjacent lobule pairs whose shared boundary has minimum
            smoothed-PV above this percentile are merged as a single lobule).
            None (default) falls back to an Otsu threshold computed on the foreground pixels.
            Range ~15 to 55; lower values merge more aggressively.
        @param pp_thr: a percentile, despite the `_thr` suffix. The sibling percentile
            parameters all use `_pct` and this one is the exception, so a value like 65.0
            means the 65th percentile, not an intensity of 65.
            Percentile of the smoothed PP distribution treated as "a portal tract is
            present here". Used both to veto valley-merges across a real portal tract and to
            build the PP ridge. None falls back to an Otsu threshold. Only has an effect when a
            PP channel is available.
        @param pp_blend: weight of the PP field in the watershed edge map,
            ``edge_map = (1 - pp_blend) * inverted_pv + pp_blend * pp_smooth``. PP is brightest
            at the lobule boundary, so the smoothed PP field is already an edge-cost map and
            needs no transform. This matters because the PV zonation curve is flat at the
            boundary whereas the PP curve is not, so most of the boundary signal is in PP.
            0 disables PP (default); only meaningful when a PP channel is supplied.
        @param higra_preview_max_px: maximum side length (pixels) of the downsampled image
            used for the interactive Higra seeding preview (shown when *preview* is True).
            Larger values give higher-fidelity previews but are slower to recompute.
        @param boundary_smooth_sigma: Gaussian radius (pixels) for post-watershed boundary
            smoothing. 0 disables. See ``segment_lobules`` for details.
        @param boundary_smooth_valley_pct: percentile of the inverted-PV distribution above
            which a pixel is frozen during smoothing (default 90 = top 10% are valleys).
        @param boundary_smooth_valley_band_px: dilation radius (pixels) applied around frozen
            valley pixels; prevents smoothed boundaries from crossing real valleys (default 2).

        @param edge_grad_blend: weight of the gradient-magnitude term blended into the
            watershed edge map.
        @param edge_local_contrast_sigma: sigma for local-contrast normalisation of the edge
            map; 0 disables it.
        @param fused_merge_pv_pct: PV percentile gate for merging CV-seeded lobule pairs whose
            shared seam has no valley; None disables the fused-pair merge entirely.
        @param seed_tissue_ring: pixels the CV seed disc extends beyond the estimated vessel
            radius.
        @param valley_merge_sigma: smoothing sigma for the valley field used by the
            below-valley merge.

        @param vessel_pv_zone_thr: ring fraction in the PV zone above which a lumen is a CV.
        @param vessel_pp_zone_thr: ring fraction in the PP zone above which a lumen is a portal.
            Both zones are median splits of the smoothed stain, so a fraction near 0.5 is the
            neutral point. When both tests pass, the larger fraction wins and PP takes the tie.
        @param vessel_zone_sigma: Gaussian sigma used to smooth PV/PP before the median split.
            0 disables zone classification and falls back to the PV ring-mean test.
        @param vessel_min_enclosure: minimum fraction of a lumen's dilation ring that must lie
            inside tissue. Rejects concavities in the outer tissue outline, which otherwise
            register as holes and become CV seeds sitting on the tissue edge.
        @param min_cv_seed_area: CV components smaller than this are dropped from the watershed
            seed mask (but kept for portality and detection metrics).

        @param clahe_clip_limit: CLAHE clip limit applied per channel during preprocessing.
        @param clahe_tile_px: CLAHE tile size in pixels (converted to OpenCV's tile *count*
            internally). Unlike a fixed 16x16 grid this is scale-invariant across slide sizes.
        @param clahe_pp_factor: multiplier on clahe_clip_limit for the PP channel only. PP is
            weakly zonated and needs harder equalisation than PV to bring portal-tract rims out.

        @param peak_thresh_factor: multiplier on the local standard deviation for an
            adaptive peak threshold; the gate is `local_mean + factor * local_std`, not a
            floor on peak height. Pinned at 0.0, which disables the adaptive gate: non-zero
            values measured monotonically worse.
        @param contour_smooth_sigma: Gaussian sigma for label-contour smoothing.
        @param min_seed_border_dist: minimum distance (pixels) from the tissue border for
            a seed. Declared float while every sibling pixel-distance is int, and
            truncated by int() at its call site; the annotation is the odd one out, not
            the behaviour.
            Measured on the hole-filled tissue mask, so central-vein lumens are not themselves
            treated as border.
        @param bg_low_val: pixels at or below this raw value are zeroed per channel before
            flatfield/CLAHE. Inert at the default of 0; retained for real slides that need it.
        @param clahe_bg_suppress: after CLAHE, values below this are forced to 0, undoing
            background that CLAHE lifted. Inert at the default of 0.
        @param fg_min_signal_frac: minimum fraction of channels that must carry signal at a pixel
            for it to count as foreground.
        @param vessel_tophat_frac: strength of the black-hat local-contrast punch used to find
            vessel lumens, as a fraction. 0.0 disables it, and that is the correct value: a
            non-zero value such as the old 0.30 makes the 50 px kernel cross threshold on
            ordinary tissue texture, punching out most of the real tissue as vessel holes.
            Lumens are found by hole topology instead.
        @param vessel_tophat_kernel_px: kernel size for that black-hat punch. Inert while
            vessel_tophat_frac is 0, i.e. always, at the shipped defaults.
        @param vessel_max_area_px: upper area gate for a vessel candidate; larger holes are
            rejected as tears or background rather than lumens.
        @param vessel_blur_px: median-blur kernel applied before vessel detection only, kept
            separate from `ksize` because blurring destroys small lumens. 0 disables it; even
            values are incremented to the next odd size internally.
        @param lobule_diameter_px: equivalent diameter of a typical lobule in your images, in
            pixels. Supplying it rescales the 22 pixel-valued parameters from the calibration
            scale (see the class docstring); lengths linearly, areas quadratically. None means
            no rescaling. Only parameters left at their default are scaled; anything you pass
            explicitly is taken to be in your own pixel frame already.
        @param vessel_lumen_pct: how a vessel lumen is recognised in the stain projection.
            None (default) uses an exact-zero test, correct wherever lumens are encoded as
            zeros (the synthetic benchmark, and any image whose background was zeroed). Real
            scans never reach exact zero inside tissue, so the default finds almost nothing on
            them. Setting a percentile (25 is a good starting point) switches to a
            median-blurred relative threshold, which recovers the lumens on real scans. Do not
            set it for synthetic-style data: there the relative threshold carves genuinely dim
            tissue into false holes and precision collapses.
        @param vessel_portality_enclosure: a second, looser enclosure threshold used only to
            build the vessel masks handed to the portality map. The strict vessel_min_enclosure
            still governs what the watershed sees. These are separate consumers with opposite
            needs: the watershed wants few, certain lumens (every false vessel is subtracted
            from its foreground), while portality wants an anchor in as many lobules as
            possible. On the annotated slide at 0.90 with vessel_lumen_pct=25 it lifts the
            number of lobules whose own vein anchors portality, with segmentation unchanged.
            None (default) reuses the strict mask for both, i.e. current behaviour.
        @param vessel_lumen_blur_px: median-blur kernel applied before the percentile threshold,
            to stop pixel noise fragmenting lumens. Only used when vessel_lumen_pct is set.
        @param um_per_px: micrometres per pixel at the level being loaded, as an alternative to
            lobule_diameter_px. Converted through TYPICAL_LOBULE_DIAMETER_UM, which is a
            literature assumption rather than a measurement, so prefer lobule_diameter_px when
            you can measure it. Giving both raises.
        """
        self.throw_out_ratio = throw_out_ratio
        self.preview = preview
        self.confirm = confirm
        self.base_level = base_level
        self.multi_otsu = multi_otsu
        self.ksize = ksize
        self.adaptive_histonorm = adaptive_histonorm
        self.bg_low_val = int(bg_low_val)
        self.clahe_bg_suppress = int(clahe_bg_suppress)
        self.local_bg_correction = bool(local_bg_correction)
        self.local_bg_sigma = float(local_bg_sigma)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_px = int(clahe_tile_px)
        self.clahe_pp_factor = float(clahe_pp_factor)

        # foreground gating
        self.fg_min_signal_frac = float(fg_min_signal_frac)

        # vessel gating
        self.min_vessel_area_pp = min_vessel_area_pp
        self.min_vessel_area_pv = min_vessel_area_pv
        self.vessel_annulus_px = vessel_annulus_px
        self.vessel_circularity_min = vessel_circularity_min
        self.vessel_tophat_frac = float(vessel_tophat_frac)
        self.vessel_tophat_kernel_px = int(vessel_tophat_kernel_px)
        self.vessel_max_area_px = int(vessel_max_area_px)
        self.vessel_blur_px = int(vessel_blur_px)

        # zone-based vessel classifier
        self.vessel_pv_zone_thr = float(vessel_pv_zone_thr)
        self.vessel_pp_zone_thr = float(vessel_pp_zone_thr)
        self.vessel_zone_sigma = float(vessel_zone_sigma)
        self.vessel_min_enclosure = float(vessel_min_enclosure)
        self.min_cv_seed_area = int(min_cv_seed_area)

        # lobule generation
        self.min_area_px = min_area_px
        self.pp_close_radius = int(pp_close_radius)

        # Higra seeded watershed
        self.blob_sigma = float(blob_sigma)
        self.peak_min_dist = int(peak_min_dist)
        self.peak_thresh_pct = float(peak_thresh_pct)
        self.valley_sigma = float(valley_sigma)
        self.cluster_merge_sigma = float(cluster_merge_sigma)
        self.edge_grad_blend = float(edge_grad_blend)
        self.edge_local_contrast_sigma = float(edge_local_contrast_sigma)
        self.fused_merge_pv_pct = (float(fused_merge_pv_pct)
                                   if fused_merge_pv_pct is not None else None)
        self.seed_tissue_ring = int(seed_tissue_ring)
        self.valley_merge_sigma = float(valley_merge_sigma)
        self.valley_merge_thr_pct = float(valley_merge_thr_pct) if valley_merge_thr_pct is not None else None
        self.pp_thr = float(pp_thr) if pp_thr is not None else None
        self.pp_blend = float(pp_blend)
        self.peak_thresh_factor = float(peak_thresh_factor)
        self.contour_smooth_sigma = float(contour_smooth_sigma)
        self.min_seed_border_dist = float(min_seed_border_dist)
        self.higra_preview_max_px = int(higra_preview_max_px)
        self.boundary_smooth_sigma = float(boundary_smooth_sigma)
        self.boundary_smooth_valley_pct = float(boundary_smooth_valley_pct)
        self.boundary_smooth_valley_band_px = int(boundary_smooth_valley_band_px)

        # make sure channel is list for later iteration
        if isinstance(channel_selection, int):
            channel_selection = [channel_selection]
        elif channel_selection is None:
            channel_selection = []
        super().__init__(metadata, channel_selection=None)
        # Override init back to have direct info on channels to invert
        self.channels = channel_selection
        # Make lists
        if isinstance(channels_pp, int):
            channels_pp = [channels_pp]
        if isinstance(channels_pv, int):
            channels_pv = [channels_pv]
        # check that the defined channels exist in the metadata objects
        if (
                (channels_pp is not None and any(i > len(metadata) - 1 for i in channels_pp)) or
                (channels_pv is not None and any(i > len(metadata) - 1 for i in channels_pv))
        ):
            console.print(
                "At least one channel for pp or pv is greater than number of metadata objects.",
                style="error"
            )
            raise ValueError(
                "At least one channel for pp or pv is greater than number of metadata objects."
            )
        # Check that at least one of the lists is not None:
        if channels_pp is None and channels_pv is None:
            console.print("No values for channel_pp or channel_pv provided.", style="error")
            raise ValueError("At least one channel has to be set in either channel_pp or channel_pv.")
        elif channels_pp is not None and channels_pv is not None:
            # Check that channels_pp and channels_pv are different
            overlap = set(channels_pp) & set(channels_pv)
            if overlap:
                console.print(
                    f"The following IDs are used both for periportal and perivenous detection: {overlap}. This is not possible.",
                    style="error")
                raise ValueError(f"Identical elements for channels_pp and channels_pv: {overlap}")
        self.channels_pp = channels_pp
        self.channels_pv = channels_pv

        # -- optional resolution-aware rescaling ------------------------------
        # Applied last: every scalable parameter is a plain `self.x = x` above and
        # none is read during __init__, so rewriting them here is equivalent to
        # having been constructed with the scaled values.
        #
        # Only parameters left at their default are rescaled.  A value the caller
        # passed explicitly is already in the caller's own pixel frame, so scaling
        # it would be wrong.  (Passing a value that happens to equal the default is
        # indistinguishable from not passing it, and counts as not passed.)
        # Seeds edited in the interactive preview, in full-resolution (x, y).
        # None means automatic seeding; only the preview ever sets it.
        self.vessel_lumen_pct = (None if vessel_lumen_pct is None
                                 else float(vessel_lumen_pct))
        self.vessel_lumen_blur_px = int(vessel_lumen_blur_px)
        self.vessel_portality_enclosure = (
            None if vessel_portality_enclosure is None
            else float(vessel_portality_enclosure))
        self._edited_seed_points_xy = None
        self.lobule_diameter_px = lobule_diameter_px
        self.um_per_px = um_per_px
        _scale = self.scale_factor(lobule_diameter_px, um_per_px)
        if (_scale == 1.0 and int(base_level) != CALIBRATION_BASE_LEVEL):
            # Running unscaled at a coarser magnification costs weighted mIoU and
            # over-segments heavily (roughly 2.6x the lobule count at 2x scale, seed
            # counts in the thousands at 4x). This is the one case a warning can
            # actually catch: nothing was set, and the level does not match the
            # calibration level. It is not a guarantee the scale is otherwise
            # correct.
            console.print(
                f"base_level={base_level} differs from the level the shipped defaults "
                f"were calibrated at (level {CALIBRATION_BASE_LEVEL}), and no "
                f"lobule_diameter_px or um_per_px was given. Without it the pixel-valued "
                f"parameters (blob_sigma, min_area_px, ...) are almost certainly wrong "
                f"for this magnification - expect heavy over-segmentation or, at a "
                f"coarser level than 3, a seed count in the thousands. Pass "
                f"lobule_diameter_px (the equivalent diameter of a typical lobule in "
                f"your images, in pixels) to rescale automatically.",
                style="warning")
        if _scale != 1.0:
            _passed = locals()
            _sig = inspect.signature(type(self).__init__).parameters
            for _k, _v in self.scaled_defaults(lobule_diameter_px, um_per_px).items():
                if _passed[_k] == _sig[_k].default:
                    setattr(self, _k, _v)

    @staticmethod
    def scale_factor(lobule_diameter_px: Optional[float] = None,
                     um_per_px: Optional[float] = None) -> float:
        """Linear factor taking the default parameters to another magnification.

        This is a magnification ratio, not a morphology one. It assumes everything
        in your images is scaled by the same factor relative to the calibration
        images (lobules, vessel lumens, nuclei and noise grain alike), which is
        exactly what changing pyramid level or scanner resolution does.

        **What it cannot express:** a specimen whose lobules are genuinely larger
        at the same magnification. One ratio cannot carry two independent
        quantities, and applying the lobule ratio to vessel and noise parameters
        would then be wrong, since a bigger lobule does not imply a bigger central
        vein or a coarser noise grain. For that case pass no scale and override the
        lobule-scale parameters directly: ``blob_sigma``, ``peak_min_dist``,
        ``valley_sigma``, ``valley_merge_sigma``, ``boundary_smooth_sigma``,
        ``boundary_smooth_valley_band_px``, ``contour_smooth_sigma``,
        ``edge_local_contrast_sigma``, ``local_bg_sigma`` and ``min_area_px``.
        ``examples/synthetic_pipeline.py`` does this, because its generator holds
        vessel size fixed while the lobule size varies.

        Give one of:

        lobule_diameter_px
            Equivalent diameter of a typical lobule in your images, in pixels,
            ``2*sqrt(median_lobule_area/pi)``. This is the honest input: it is the
            same quantity the defaults are anchored to, measured the same way.
        um_per_px
            Micrometres per pixel at the level you are loading. Converted via
            :data:`TYPICAL_LOBULE_DIAMETER_UM`, which is an assumption rather than a
            measurement, so prefer *lobule_diameter_px* when you can measure it.

        Returns 1.0 when neither is given, which is what keeps the operator's
        behaviour unchanged by default.
        """
        if lobule_diameter_px is None and um_per_px is None:
            return 1.0
        if lobule_diameter_px is not None and um_per_px is not None:
            raise ValueError("Give lobule_diameter_px or um_per_px, not both.")
        if um_per_px is not None:
            if float(um_per_px) <= 0:
                raise ValueError(f"um_per_px must be > 0, got {um_per_px}")
            lobule_diameter_px = TYPICAL_LOBULE_DIAMETER_UM / float(um_per_px)
        if float(lobule_diameter_px) <= 0:
            raise ValueError(
                f"lobule_diameter_px must be > 0, got {lobule_diameter_px}")
        return float(lobule_diameter_px) / CALIBRATION_LOBULE_DIAMETER_PX

    @classmethod
    def scaled_defaults(cls, lobule_diameter_px: Optional[float] = None,
                        um_per_px: Optional[float] = None) -> Dict[str, Any]:
        """The pixel-valued defaults rescaled to another magnification.

        Lengths scale by :meth:`scale_factor`, areas by its square.  Useful on its
        own for inspecting what a scale change implies, or to pass through::

            LobuleSegmentor(mds, **LobuleSegmentor.scaled_defaults(um_per_px=2.2))
        """
        s = cls.scale_factor(lobule_diameter_px, um_per_px)
        sig = inspect.signature(cls.__init__).parameters
        out: Dict[str, Any] = {}
        for k in _SCALE_LENGTH:
            v = sig[k].default
            out[k] = int(round(v * s)) if isinstance(v, int) else float(v) * s
        for k in _SCALE_AREA:
            out[k] = int(round(sig[k].default * s * s))
        return out


    def _pv_index(self) -> int:
        """Index of the PV (perivenous) channel in the working stack."""
        return int(self.channels_pv[0]) if self.channels_pv else 0

    def _pp_index(self) -> int:
        """Index of the PP (periportal) channel, or 1 when there is none.

        The shared vessel/preprocessing helpers default to 0/1 because the
        synthetic benchmark always uses that layout.  Production stains can sit
        at any index, so the indices are passed explicitly rather than assumed.
        """
        return int(self.channels_pp[0]) if self.channels_pp else 1

    def _stain_indices(self, n_channels: int) -> list:
        """Channels selected as PV or PP, in order, de-duplicated.

        These are the only channels that should shape the watershed. Structural
        channels still help find tissue at load time, but letting them into the
        vessel detector's foreground projection hides lumens: a nuclear or
        membrane channel that is bright inside a lumen keeps that lumen
        "foreground", so the enclosed-hole test never finds it. Measured on a
        real annotated slide, restricting the projection took vessel candidates
        inside the size gates from 7 to 16.
        """
        idx = list(self.channels_pv or []) + list(self.channels_pp or [])
        idx = [int(c) for c in dict.fromkeys(idx) if 0 <= int(c) < int(n_channels)]
        return idx or [0]

    def _combine_channels(self, stack, channels):
        """Max-project the selected channels into one stain image.

        Previously only ``channels[0]`` was used, so passing a list quietly
        discarded the rest - ``channels_pv=[0, 1]`` in the simliva example meant
        CYP3A4 never reached the watershed. Max is the rule already used
        elsewhere here to combine channels into a projection.
        """
        chans = [int(c) for c in (channels or []) if 0 <= int(c) < stack.shape[2]]
        if not chans:
            chans = [0]
        if len(chans) == 1:
            return stack[:, :, chans[0]].astype(np.float32)
        return stack[:, :, chans].max(axis=2).astype(np.float32)

    @staticmethod
    def _widget_range(current: float, hard_min: float, hard_max: float,
                      shrink: float = 4.0, grow: float = 4.0):
        """Slider bounds that always contain *current*, a value that may have
        been rescaled by `lobule_diameter_px` / `um_per_px`.

        A previewed pixel-valued parameter is scaled at construction time, but
        every interactive slider bounded it with a fixed literal. At the
        calibration scale that literal happened to contain the default; at
        another scale it does not, and magicgui raises ValueError building the
        widget, which is what this replaces.

        Only stretches a bound the current value actually falls outside of; a
        value already inside [hard_min, hard_max] returns that pair unchanged.
        Growing unconditionally (e.g. always current*4) would widen widgets even
        at the calibration scale, changing behaviour for the common case for no
        reason: vessel_zone_sigma's default 40 against hard_max 120 would give
        (0, 160) rather than the existing (0, 120).
        """
        lo = hard_min if current >= hard_min else current / shrink
        hi = hard_max if current <= hard_max else current * grow
        return lo, hi

    def _has_pp_channel(self, n_channels: int) -> bool:
        """True when a PP channel is configured and present in the stack."""
        return bool(self.channels_pp) and self._pp_index() < int(n_channels)

    def _preview_zone_vessels(self, image_stack_u8, run_detector):
        """Interactive tuning of the zone-based vessel detector.

        Only `detect_and_classify_vessels` re-runs on each update (a few
        seconds), not the whole pipeline, so this stays responsive.

        Returns the final (cv_mask, pf_mask, tissue_mask, vessel_holes).
        """
        import napari
        from magicgui import magicgui

        state = dict(zip(("cv", "pf", "tis", "holes"), run_detector()))

        pending = dict(
            vessel_pv_zone_thr=float(self.vessel_pv_zone_thr),
            vessel_pp_zone_thr=float(self.vessel_pp_zone_thr),
            vessel_zone_sigma=float(self.vessel_zone_sigma),
            vessel_min_enclosure=float(self.vessel_min_enclosure),
            min_vessel_area_pp=int(self.min_vessel_area_pp),
            min_vessel_area_pv=int(self.min_vessel_area_pv),
            vessel_circularity_min=float(self.vessel_circularity_min),
            vessel_max_area_px=int(self.vessel_max_area_px),
            vessel_annulus_px=int(self.vessel_annulus_px),
            min_cv_seed_area=int(self.min_cv_seed_area),
        )
        defaults = dict(pending)

        def _apply_pending() -> None:
            for k, v in pending.items():
                setattr(self, k, v)

        def _base_gray() -> np.ndarray:
            base = image_stack_u8[:, :, self._pv_index()].astype(np.float32)
            nz = base[base > 0]
            p99 = float(np.percentile(nz, 99)) if nz.size else 255.0
            return (np.clip(base / max(p99, 1e-6), 0, 1) * 255).astype(np.uint8)

        def _zone_masks():
            """The PV/PP zones the classifier actually decides on.

            Recomputed here with the same smoothing and median split that
            detect_and_classify_vessels uses, so what is displayed is the
            evidence behind vessel_pv_zone_thr / vessel_pp_zone_thr rather than
            just their outcome. Previously these were never shown, so those two
            thresholds had to be tuned blind.
            """
            tis = state["tis"]
            fg = tis & ~state["holes"]
            sig = float(self.vessel_zone_sigma)
            blank = np.zeros_like(tis)
            has_pp = self._has_pp_channel(image_stack_u8.shape[-1])
            if sig <= 0 or not fg.any():
                return blank, blank, has_pp
            # Same inputs the detector uses: the median-blurred stack, not raw.
            _k = int(self.vessel_blur_px)
            if _k % 2 == 0:
                _k += 1
            _src = (np.stack([cv2.medianBlur(image_stack_u8[:, :, c], _k)
                              for c in range(image_stack_u8.shape[2])], axis=-1)
                    if _k >= 3 else image_stack_u8)
            # PV zone: pericentral tissue = high PV. Uses the max over the selected
            # PV channels (what the classifier reads), not just channels_pv[0].
            pv_c = self._combine_channels(_src, self.channels_pv).astype(np.float32)
            pv_s = _gaussian_filter(pv_c, sig)
            pv_zone = (pv_s > float(np.median(pv_s[fg]))) & fg
            if has_pp:
                pp_c = self._combine_channels(_src, self.channels_pp).astype(np.float32)
                pp_s = _gaussian_filter(pp_c, sig)
                pp_zone = (pp_s > float(np.median(pp_s[fg]))) & fg
            else:
                # No PP marker: the periportal zone is the tissue that is not
                # pericentral, i.e. the complement of the PV zone (low PV). Derived
                # from the PV split, not a real PP-marker median split.
                pp_zone = fg & ~pv_zone
            return pv_zone, pp_zone, has_pp

        viewer = napari.Viewer()
        # Separate toggleable layers rather than one pre-composited RGB image, so
        # each piece of evidence can be turned on and off and its opacity changed
        # while tuning the threshold that produces it.
        viewer.add_image(_base_gray(), name="PV channel",
                         colormap="gray", blending="translucent")
        _pv_zone0, _pp_zone0, _has_pp = _zone_masks()
        if not _has_pp:
            console.print(
                "No PP marker: the PP zone is derived as the periportal complement "
                "of the PV zone (low PV = periportal). Pass a periportal channel as "
                "channels_pp for a real PP median split.", style="info")
        _seed0 = filter_cv_seeds(state["cv"], self.min_cv_seed_area)
        # All mask layers pin contrast_limits=[0, 1]. Without it a 0/1 uint8 image
        # can auto-scale against the dtype range and render a value of 1 at ~1/255
        # brightness, effectively invisible, which reads as nothing detected
        # even though the mask is populated. (The hole layer uses add_labels, which
        # is why it stays visible while the others vanished.)
        zone_pv_layer = viewer.add_image(
            _pv_zone0.astype(np.uint8), name="PV zone (pericentral)",
            colormap="cyan", opacity=0.40, blending="translucent",
            contrast_limits=[0, 1], visible=True)
        zone_pp_layer = viewer.add_image(
            _pp_zone0.astype(np.uint8),
            name=("PP zone (median split)" if _has_pp
                  else "PP zone (periportal complement)"),
            colormap="magenta", opacity=0.40, blending="translucent",
            contrast_limits=[0, 1], visible=True)
        holes_layer = viewer.add_labels(
            state["holes"].astype(np.uint8), name="hole candidates", opacity=0.35)
        # CV split into seeding vs too-small, so min_cv_seed_area's effect is visible
        cv_layer = viewer.add_image(
            _seed0.astype(np.uint8), name="central veins (used as seeds)",
            colormap="yellow", opacity=0.90, blending="additive",
            contrast_limits=[0, 1])
        cv_small_layer = viewer.add_image(
            (state["cv"] & ~_seed0).astype(np.uint8),
            name="central veins (below min seed area)",
            colormap="bop orange", opacity=0.70, blending="additive",
            contrast_limits=[0, 1])
        pf_layer = viewer.add_image(
            state["pf"].astype(np.uint8), name="portal vessels",
            colormap="red", opacity=0.90, blending="additive",
            contrast_limits=[0, 1])

        # Bounds for the pixel-valued widgets below, widened to always contain the
        # current (possibly lobule_diameter_px-rescaled) value, see _widget_range.
        # At the calibration scale these equal the literals that used to be
        # hardcoded; only a rescaled run stretches them.
        def _int_range(*a, **kw):
            lo, hi = self._widget_range(*a, **kw)
            return int(lo), int(hi)
        _r_zone_sigma = self._widget_range(defaults["vessel_zone_sigma"], 0.0, 120.0)
        _r_area_pp = _int_range(defaults["min_vessel_area_pp"] or 1, 0, 5000, grow=16.0)
        _r_area_pv = _int_range(defaults["min_vessel_area_pv"] or 1, 0, 5000, grow=16.0)
        _r_max_area = _int_range(defaults["vessel_max_area_px"], 500, 60000, grow=16.0)
        _r_annulus = _int_range(defaults["vessel_annulus_px"], 1, 30)
        _r_seed_area = _int_range(defaults["min_cv_seed_area"] or 1, 0, 1000, grow=16.0)

        @magicgui(
            layout="vertical",
            auto_call=True,
            vessel_pv_zone_thr={"min": 0.0, "max": 1.0, "step": 0.05,
                                "label": "PV zone threshold",
                                "tooltip": "Ring fraction in the PV zone above which a lumen is a "
                                           "central vein. The zone is a median split, so ~0.5 is neutral."},
            vessel_pp_zone_thr={"min": 0.0, "max": 1.0, "step": 0.05,
                                "label": "PP zone threshold",
                                "tooltip": "Ring fraction in the PP zone above which a lumen is a portal."},
            vessel_zone_sigma={"min": _r_zone_sigma[0], "max": _r_zone_sigma[1], "step": 5.0,
                               "label": "Zone sigma (px)",
                               "tooltip": "Smoothing of PV/PP before the median split. 0 disables "
                                          "zone classification and falls back to the ring-mean test."},
            vessel_min_enclosure={"min": 0.0, "max": 1.0, "step": 0.05,
                                  "label": "Min enclosure",
                                  "tooltip": "Fraction of a lumen's ring that must lie inside tissue. "
                                             "Rejects concavities in the outer outline."},
            min_vessel_area_pp={"min": _r_area_pp[0], "max": _r_area_pp[1], "step": 25,
                                "label": "Portal: min detection area (px²)",
                                "tooltip": "A hole smaller than this is never classified as a "
                                           "portal vessel. Acts at detection, before any "
                                           "seeding. Raise to reject small debris."},
            min_vessel_area_pv={"min": _r_area_pv[0], "max": _r_area_pv[1], "step": 25,
                                "label": "CV: min detection area (px²)",
                                "tooltip": "A hole smaller than this is never classified as a "
                                           "central vein. Acts at detection, distinct from "
                                           "'CV: min seed area' below, which decides which "
                                           "already-detected veins are allowed to seed the "
                                           "watershed. Raise to reject small debris."},
            vessel_circularity_min={"min": 0.0, "max": 1.0, "step": 0.02,
                                    "label": "Min circularity",
                                    "tooltip": "4*pi*area/perimeter^2; 1.0 is a perfect circle. "
                                               "Rejects ragged, non-lumen-shaped holes. Lower "
                                               "to keep irregular vessel cuts."},
            vessel_max_area_px={"min": _r_max_area[0], "max": _r_max_area[1], "step": 500,
                                "label": "Max vessel area (px²)",
                                "tooltip": "Upper area gate: larger holes are treated as tears "
                                           "or background, not lumens. Raise if genuine "
                                           "large-bore vessel cuts are being rejected."},
            vessel_annulus_px={"min": _r_annulus[0], "max": _r_annulus[1],
                               "label": "Annulus width (px)",
                               "tooltip": "Width of the ring dilated around each hole. The ring "
                                          "supplies both the enclosure test and the PV/PP "
                                          "fractions used to classify CV vs portal."},
            min_cv_seed_area={"min": _r_seed_area[0], "max": _r_seed_area[1], "step": 25,
                              "label": "CV: min seed area (px²)",
                              "tooltip": "Acts at SEEDING, after detection: central veins "
                                         "smaller than this are still detected (and still used "
                                         "for portality) but do not seed the watershed. Shown "
                                         "in the 'below min seed area' layer. 0 disables it."},
        )
        def controls(
            vessel_pv_zone_thr: float = defaults["vessel_pv_zone_thr"],
            vessel_pp_zone_thr: float = defaults["vessel_pp_zone_thr"],
            vessel_zone_sigma: float = defaults["vessel_zone_sigma"],
            vessel_min_enclosure: float = defaults["vessel_min_enclosure"],
            min_vessel_area_pp: int = defaults["min_vessel_area_pp"],
            min_vessel_area_pv: int = defaults["min_vessel_area_pv"],
            vessel_circularity_min: float = defaults["vessel_circularity_min"],
            vessel_max_area_px: int = defaults["vessel_max_area_px"],
            vessel_annulus_px: int = defaults["vessel_annulus_px"],
            min_cv_seed_area: int = defaults["min_cv_seed_area"],
        ):
            pending.update(
                vessel_pv_zone_thr=float(vessel_pv_zone_thr),
                vessel_pp_zone_thr=float(vessel_pp_zone_thr),
                vessel_zone_sigma=float(vessel_zone_sigma),
                vessel_min_enclosure=float(vessel_min_enclosure),
                min_vessel_area_pp=int(min_vessel_area_pp),
                min_vessel_area_pv=int(min_vessel_area_pv),
                vessel_circularity_min=float(vessel_circularity_min),
                vessel_max_area_px=int(vessel_max_area_px),
                vessel_annulus_px=int(vessel_annulus_px),
                min_cv_seed_area=int(min_cv_seed_area),
            )

        def _recompute() -> None:
            _apply_pending()
            state.update(zip(("cv", "pf", "tis", "holes"), run_detector()))
            _n_cv, _ = _diag_label_for_preview(state["cv"])
            _n_pf, _ = _diag_label_for_preview(state["pf"])
            console.print(
                f"Vessels: {_n_cv} central, {_n_pf} portal  "
                f"(pv_thr={pending['vessel_pv_zone_thr']:.2f} "
                f"pp_thr={pending['vessel_pp_zone_thr']:.2f} "
                f"sigma={pending['vessel_zone_sigma']:.0f} "
                f"encl={pending['vessel_min_enclosure']:.2f})",
                style="info",
            )
            _pvz, _ppz, _ = _zone_masks()
            zone_pv_layer.data = _pvz.astype(np.uint8)
            zone_pp_layer.data = _ppz.astype(np.uint8)
            holes_layer.data = state["holes"].astype(np.uint8)
            _seed = filter_cv_seeds(state["cv"], self.min_cv_seed_area)
            cv_layer.data = _seed.astype(np.uint8)
            cv_small_layer.data = (state["cv"] & ~_seed).astype(np.uint8)
            pf_layer.data = state["pf"].astype(np.uint8)

        def _reset() -> None:
            pending.update(defaults)
            for k, v in defaults.items():
                getattr(controls, k).value = v
            _recompute()

        # Pack the controls to the TOP of the right dock instead of letting Qt
        # centre them vertically (which leaves a large gap above the panel).
        try:
            controls.native.layout().addStretch(1)
        except Exception:
            pass
        viewer.window.add_dock_widget(controls, area="right",
                                      name="Zone vessel detection")
        action = run_napari_preview_action(
            viewer, controls,
            require_confirm=self.confirm,
            on_update=_recompute,
            on_reset=_reset,
            on_confirm=_apply_pending,
            on_apply_values=_apply_pending,
            on_back=_apply_pending,
        )
        if action == "abort":
            raise PipelineAbort()
        if action == "back":
            raise PipelineBack(1)
        _apply_pending()
        return state["cv"], state["pf"], state["tis"], state["holes"]

    def _correct_stack(self, image_stack: np.ndarray,
                       tissue_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Flatfield + CLAHE correction, per channel. Returns float32 (H, W, N).

        This is the first half of preprocessing. It is separate from :meth:`_filter`
        because the two consumers need *different* stages of the chain: vessel
        detection runs on this corrected stack directly, while the PV/PP stains fed
        to ``segment_lobules`` get the additional blur + p99 normalisation that
        :meth:`_filter` appends. Blurring before vessel detection destroys small
        lumens, which is why the split exists.

        Delegates to ``vessel_detection.preprocess_stack`` so every caller runs
        the identical operation.
        """
        if image_stack.ndim != 3:
            raise ValueError(f"Expected (H, W, N) stack, got {image_stack.shape}")

        stack = image_stack
        if int(self.bg_low_val) > 0:
            # Per-channel background suppression. A previous implementation used a
            # shared mask across all channels (np.any(stack <= bg_low_val, axis=-1)),
            # which leaked spatial patterns from zonated channels (e.g. CYP) into
            # uniform ones (e.g. DAPI), creating false zonation artifacts.
            # Inert at the default of 0; retained for real slides that need it.
            stack = stack.copy()
            stack[stack <= int(self.bg_low_val)] = 0

        if tissue_mask is None:
            tissue_mask = stack.max(axis=-1) > 0

        _sigma = float(self.local_bg_sigma) if (
            self.local_bg_correction and self.local_bg_sigma > 0) else 0.0
        _clip = float(self.clahe_clip_limit) if self.adaptive_histonorm else 0.0

        corrected = preprocess_stack(
            stack,
            local_bg_sigma=_sigma,
            clahe_clip_limit=_clip,
            clahe_tile_px=int(self.clahe_tile_px),
            clahe_pp_factor=float(self.clahe_pp_factor),
            has_pp=self._has_pp_channel(stack.shape[-1]),
            tissue_mask=tissue_mask,
            pp_index=self._pp_index(),
        )

        if self.adaptive_histonorm and int(self.clahe_bg_suppress) > 0:
            # Suppress background lifted by CLAHE. Inert at the default of 0;
            # retained for real slides that need it.
            corrected = np.where(corrected < float(self.clahe_bg_suppress),
                                 0.0, corrected)
        return corrected.astype(np.float32)

    def _filter(self, image_stack: np.ndarray,
                tissue_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Full preprocessing chain. Keeps dtype uint8, returns (H, W, N).

        Flatfield + CLAHE (:meth:`_correct_stack`), then the capped median blur and
        p99 normalisation from ``vessel_detection.stains_from_stack``. Used by the
        interactive preprocessing preview; :meth:`apply` calls the two halves
        separately so vessel detection sees the unblurred stack.
        """
        return self._finalize_stains(
            self._correct_stack(image_stack, tissue_mask=tissue_mask))

    def _finalize_stains(self, corrected: np.ndarray) -> np.ndarray:
        """Second half of preprocessing: capped median blur + p99 normalisation.

        Takes the float32 output of :meth:`_correct_stack` and returns uint8.
        Shares ``vessel_detection.stains_from_stack``'s arithmetic; that helper
        returns the split PV/PP pair, this one keeps the full stack.
        """
        out = np.clip(corrected, 0, 255).astype(np.uint8)
        for c in range(out.shape[-1]):
            ch = out[..., c]
            k = int(self.ksize)
            if k % 2 == 0:
                k += 1
            # Capped at 15 px: PV peaks sit next to central veins only 16-48 px
            # across, and an unclamped 25 px kernel blurs them away.
            k = min(k, 15)
            if k >= 3:
                ch = cv2.medianBlur(ch, k)
            ch = cv2.medianBlur(ch, 3)
            # Normalise to the 99th percentile; >0 avoids zero-inflation from
            # background.
            nz = ch[ch > 0]
            p99 = float(np.percentile(nz, 99)) if nz.size else 255.0
            if p99 <= 0:
                p99 = 1.0
            out[..., c] = (ch.astype(np.float32) / p99 * 255.0).clip(0, 255).astype(np.uint8)
        return out

    def _load_stack_only(self, level: int) -> np.ndarray:
        """
        Load and per-channel normalize the image stack without applying the
        tissue mask, cropping, or channel inversion.

        Used by the napari preprocessing preview so the "raw" layers reflect
        the actual data on disk rather than the post-masked, post-cropped stack
        that _load_and_invert_images_from_metadatas returns.
        """
        lvl = int(level)
        arrays = []
        for i in range(len(self.metadata)):
            img = np.asarray(load_level_from_multiscale(self.load_image(i), lvl))
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = np.mean(img[..., :3], axis=-1).astype(img.dtype)
            arrays.append(img)
        h_min = min(a.shape[0] for a in arrays)
        w_min = min(a.shape[1] for a in arrays)
        arrays = [a[:h_min, :w_min] for a in arrays]
        arrays = [minmax_to_uint8(a) for a in arrays]
        return np.dstack(arrays)

    def _interactive_preprocessing_preview(
            self,
            report_path: Optional[Path] = None,
            *,
            require_confirm: bool = False,
            return_action: bool = False,
    ) -> Union[bool, str]:
        """
        Interactive napari preview for preprocessing (ROI crop + filter) before segmentation.

        Button-only behavior: parameter widgets only stage values, and computation
        and layer updates happen only when pressing "Preview / Update" (or
        Confirm/Back, which apply once before closing).

        Returns
        -------
        Union[bool, str]
            bool (legacy): True if confirmed (or if require_confirm=False), False if not confirmed.
            str (wizard): "confirm", "back", or "closed".
        """

        # Load multiscale handles once; discover common levels from these handles.
        multiscales = [self.load_image(i) for i in range(len(self.metadata))]
        common_levels = common_pyramid_levels(multiscales)
        # Use the user-specified base_level if it exists in the pyramid,
        # otherwise fall back to the automatic choice.
        if self.base_level in common_levels:
            default_level = self.base_level
        else:
            default_level = choose_default_preview_level(common_levels)

        def _title_for_md(md: Metadata, idx: int) -> str:
            # Prefer stain name (more meaningful in napari layer list)
            stains = getattr(md, "stains", None)
            if stains and isinstance(stains, dict):
                first = stains.get(0, None)
                if first:
                    return str(first)
            p = getattr(md, "path_original", None) or getattr(md, "path", None) or getattr(md, "name", None)
            if isinstance(p, tuple) and p:
                p = p[0]
            if p is not None:
                try:
                    return Path(p).stem
                except Exception:
                    return str(p)
            return f"image_{idx}"

        titles = [_title_for_md(md, i) for i, md in enumerate(self.metadata)]

        # Cache raw (downsampled) stacks per level; filtered stacks per
        # (level, bg_low_val, clahe_bg_suppress, local_bg_sigma, local_bg_correction).
        raw_cache: Dict[int, np.ndarray] = {}
        filt_cache: Dict[Tuple[int, int, int, int, bool], np.ndarray] = {}

        def _get_raw_stack(level: int) -> np.ndarray:
            lvl = int(level)
            if lvl in raw_cache:
                return raw_cache[lvl]

            # Load without tissue masking or cropping so the napari "raw" layers
            # reflect the actual per-channel data from disk.
            raw_small = downsample_stack_to_max_side(self._load_stack_only(lvl), 2048)
            raw_cache[lvl] = raw_small
            return raw_small

        def _get_filtered_stack(level: int, bg_low_val: int, clahe_bg_suppress: int,
                                local_bg_sigma: float, local_bg_correction: bool) -> np.ndarray:
            # Always sync instance state to the UI values, even if we hit the cache.
            self.bg_low_val = int(bg_low_val)
            self.clahe_bg_suppress = int(clahe_bg_suppress)
            self.local_bg_sigma = float(local_bg_sigma)
            self.local_bg_correction = bool(local_bg_correction)

            key = (int(level), int(bg_low_val), int(clahe_bg_suppress),
                   int(local_bg_sigma), bool(local_bg_correction))
            if key in filt_cache:
                return filt_cache[key]

            raw = _get_raw_stack(level)
            filt = self._filter(raw)
            filt_cache[key] = filt
            return filt

        import napari
        from magicgui import magicgui
        viewer = napari.Viewer()

        raw0 = _get_raw_stack(default_level)
        filt0 = _get_filtered_stack(default_level, self.bg_low_val, self.clahe_bg_suppress,
                                    self.local_bg_sigma, self.local_bg_correction)

        raw_mean_layer = viewer.add_image(
            raw0.mean(axis=2).astype(np.uint8),
            name="raw/mean",
            colormap="gray",
        )

        def _compute_tissue_mask(raw: np.ndarray) -> np.ndarray:
            """Compute tissue mask from the raw stack.

            Since detect_tissue_mask_multiotsu now uses fraction-based
            parameters, the result is resolution-invariant and can be
            computed directly on the downsampled preview stack.
            """
            gray = ensure_grayscale_uint8(raw)
            if self.multi_otsu is False:
                return detect_tissue_mask(gray, morphological_radius=5)
            return detect_tissue_mask_multiotsu(
                gray, auto=(self.multi_otsu is None),
            )

        raw_layers: List[Any] = []
        filt_layers: List[Any] = []
        for c, title in enumerate(titles):
            raw_layers.append(
                viewer.add_image(
                    raw0[..., c],
                    name=f"raw/{title}",
                    colormap="gray",
                    visible=False,
                )
            )
            filt_layers.append(
                viewer.add_image(
                    filt0[..., c],
                    name=f"filtered/{title}",
                    colormap="gray",
                )
            )

        # Tissue mask overlay: shows what will be kept (True) vs removed (False)
        tissue0 = _compute_tissue_mask(raw0)
        tissue_layer = viewer.add_labels(
            tissue0.astype(np.int8),
            name="tissue mask",
            opacity=0.3,
            visible=False,
        )

        defaults = {
            "level": int(default_level),
            "bg": int(self.bg_low_val),
            "clahe": int(self.clahe_bg_suppress),
            "fg_frac": float(self.fg_min_signal_frac),
            "ff_sigma": float(self.local_bg_sigma),
            "ff_on": bool(self.local_bg_correction),
        }

        pending = dict(defaults)

        def _apply_update() -> None:
            lvl = int(pending["level"])
            bg = int(pending["bg"])
            clahe = int(pending["clahe"])
            self.base_level = lvl
            self.bg_low_val = bg
            self.clahe_bg_suppress = clahe
            self.fg_min_signal_frac = float(pending["fg_frac"])

            raw = _get_raw_stack(lvl)
            filt = _get_filtered_stack(lvl, bg, clahe,
                                       float(pending["ff_sigma"]), bool(pending["ff_on"]))

            raw_mean_layer.data = raw.mean(axis=2).astype(np.uint8)
            for c in range(raw.shape[2]):
                raw_layers[c].data = raw[..., c]
                filt_layers[c].data = filt[..., c]
            tissue_layer.data = _compute_tissue_mask(raw).astype(np.int8)

        # See _widget_range: keeps local_bg_sigma's slider containing the CURRENT
        # (possibly rescaled) value rather than a fixed literal it can exceed.
        _r_bg_sigma_lo, _r_bg_sigma_hi = self._widget_range(
            float(self.local_bg_sigma), 0.0, 1500.0, grow=8.0)

        @magicgui(
            layout="vertical",
            auto_call=True,  # stages values only; NO computation here
            preview_level={"choices": common_levels},
            bg_low_val={"min": 0, "max": 255, "step": 1,
                        "label": "Background cut (raw value)",
                        "tooltip": "Pixels at or below this raw value are zeroed per channel "
                                   "before flatfield/CLAHE. 0 disables it. For slides whose "
                                   "background sits above zero."},
            clahe_bg_suppress={"min": 0, "max": 255, "step": 1,
                               "label": "Post-CLAHE background cut",
                               "tooltip": "After CLAHE, values below this are forced to 0, "
                                          "undoing background that CLAHE lifted. 0 disables it."},
            fg_min_signal_frac={"min": 0.0, "max": 1.0, "step": 0.01,
                                "label": "Min signal fraction for foreground",
                                "tooltip": "Fraction of channels that must carry signal at a "
                                           "pixel for it to count as foreground."},
            local_bg_sigma={"min": _r_bg_sigma_lo, "max": _r_bg_sigma_hi, "step": 25,
                            "label": "Flatfield sigma (px)",
                            "tooltip": "Gaussian radius for the pseudo-flatfield background "
                                       "estimate. Set it comparable to or below the shading "
                                       "scale, roughly one lobule width. Too large and it "
                                       "stops correcting; too small and it eats the zonation."},
        )
        def controls(
                preview_level: int = int(default_level),
                bg_low_val: int = int(self.bg_low_val),
                clahe_bg_suppress: int = int(self.clahe_bg_suppress),
                fg_min_signal_frac: float = float(self.fg_min_signal_frac),
                local_bg_sigma: float = float(self.local_bg_sigma),
                local_bg_correction: bool = bool(self.local_bg_correction),
        ):
            pending["level"] = int(preview_level)
            pending["bg"] = int(bg_low_val)
            pending["clahe"] = int(clahe_bg_suppress)
            pending["fg_frac"] = float(fg_min_signal_frac)
            pending["ff_sigma"] = float(local_bg_sigma)
            pending["ff_on"] = bool(local_bg_correction)

        def on_update() -> None:
            _apply_update()

        def on_reset() -> None:
            pending.update(defaults)

            controls.preview_level.value = int(defaults["level"])
            controls.bg_low_val.value = int(defaults["bg"])
            controls.clahe_bg_suppress.value = int(defaults["clahe"])
            controls.fg_min_signal_frac.value = float(defaults["fg_frac"])
            controls.local_bg_sigma.value = float(defaults["ff_sigma"])
            controls.local_bg_correction.value = bool(defaults["ff_on"])

            _apply_update()

        def on_confirm() -> None:
            # Keep self.* as-is (last Recalculate or initial values).
            # The user is confirming what they see, not unseen slider changes.
            pass

        def on_apply_values() -> None:
            _apply_update()

        def on_back() -> None:
            _apply_update()

        action = run_napari_preview_action(
            viewer,
            controls,
            require_confirm=bool(require_confirm),
            on_update=on_update,
            on_reset=on_reset,
            on_confirm=on_confirm,
            on_apply_values=on_apply_values,
            on_back=on_back,
        )

        if return_action:
            # Both "confirm" and "apply_values" proceed forward
            if action == "apply_values":
                return "confirm"
            return action

        return True if not require_confirm else (action in ("confirm", "apply_values"))

    def _load_and_invert_images_from_metadatas(
            self,
            report_path: Optional[Path] = None,
            level: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], Dict[int, Any]]:
        """
        Load each image at the pyramid level defined, invert selected channels,
        harmonize shapes, and stack along the last axis -> (H, W, N).

        Returns (stack, fg_maxproj, bbox, orig_shapes) where fg_maxproj is a
        float32 (H, W) max-projection computed from the original-bitdepth data
        before uint8 quantization and before the Otsu tissue mask.
        """

        lvl = int(self.base_level if level is None else level)

        arrays: List[np.ndarray] = []
        for i, _md in enumerate(self.metadata):
            try:
                img = load_level_from_multiscale(self.load_image(i), lvl)
            except Exception as e:
                raise Exception(f"Level {lvl} failed for image {i} in stack") from e

            img = np.asarray(img)
            # ensure 2D
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = np.mean(img[..., :3], axis=-1).astype(img.dtype)

            arrays.append(img)

        if not arrays:
            raise RuntimeError("No images loaded after filtering.")

        # harmonize shapes before stacking
        h_min = min(a.shape[0] for a in arrays)
        w_min = min(a.shape[1] for a in arrays)
        arrays = [a[:h_min, :w_min] for a in arrays]

        # Compute a float32 max-projection before uint8 quantization.
        # minmax_to_uint8 maps dim-but-real tissue signals (e.g. 50/50000 in
        # 16-bit E-Cadherin) to 0 in uint8, making them indistinguishable from
        # true background. The float max-projection preserves these values.
        arrays_f32 = [a.astype(np.float32) for a in arrays]
        eps = 1e-10
        arrays_norm = [
            (af - float(np.nanmin(af))) / (float(np.nanmax(af)) - float(np.nanmin(af)) + eps)
            for af in arrays_f32
        ]
        fg_maxproj = np.dstack(arrays_norm).max(axis=2)  # (H, W) float32, [0..1]

        # Normalize each channel independently to [0, 255] before stacking.
        # This prevents high-intensity channels from dominating the tissue-detection
        # grayscale (e.g. a bright zonated CYP channel biasing the mean against DAPI),
        # and avoids silent bit-depth truncation for >8-bit source images.
        arrays = [minmax_to_uint8(a) for a in arrays]
        stack = np.dstack(arrays)

        # Get resolutions / image sizes for later back mapping based on stain 0
        orig_shapes = discover_pyramid_shapes(self.load_image(0))

        # ROI detection block
        console.print("Complete. Detecting ROI...", style="info")
        stack_grayscale = ensure_grayscale_uint8(stack)
        if self.multi_otsu is False:
            # Explicit 2-class override: simple Otsu, no microscopy BG handling.
            tissue_mask = detect_tissue_mask(stack_grayscale, morphological_radius=5)
        else:
            # multi_otsu is None (auto) or True (force 3-class legacy).
            # auto=True  -> decide bimodal/trimodal from mic_bg_frac heuristic.
            # auto=False -> always tissue = top class only (old multi_otsu=True behaviour).
            tissue_mask = detect_tissue_mask_multiotsu(
                stack_grayscale,
                auto=(self.multi_otsu is None),
                report_path=report_path,
            )

        # stack: (H,W,C); tissue_mask: (H,W) True=tissue
        stack[~tissue_mask] = 0

        # crop to tissue bbox
        bbox = largest_bbox(tissue_mask.astype(np.uint8))
        stack = crop_image(stack, bbox)
        # Crop the float32 max-projection with the same bbox.
        bx_raw, by_raw, bw_raw, bh_raw = bbox
        fg_maxproj[~tissue_mask] = 0.0
        fg_maxproj = fg_maxproj[by_raw:by_raw + bh_raw, bx_raw:bx_raw + bw_raw]
        # Convert bbox from (x, y, w, h) to (min_r, min_c, max_r, max_c)
        # for downstream consumers (build_mask_pyramid, to_base_full, portality crop).
        bx, by, bw, bh = bbox
        bbox = (by, bx, by + bh, bx + bw)

        # Inversion and discard only after cropping.
        # Note: discarding channels changes channel indices. We first decide what
        # to drop, then delete in reverse order to avoid index shifting bugs.
        drop: List[int] = []

        for i in range(stack.shape[-1]):
            # optional inversion
            if self.channels is not None and i in self.channels:
                if np.issubdtype(stack[:, :, i].dtype, np.integer):
                    info = np.iinfo(stack[:, :, i].dtype)
                    stack[:, :, i] = info.max - stack[:, :, i]
                else:  # float images assumed in [0,1]
                    stack[:, :, i] = 1.0 - stack[:, :, i]

            if self.throw_out_ratio is not None:
                non_bg = np.count_nonzero(stack[:, :, i] != 255)
                ratio = non_bg / float(stack[:, :, i].size)
                if ratio < float(self.throw_out_ratio):
                    drop.append(i)

        if drop:
            for i in sorted(drop, reverse=True):
                console.print(
                    f"Discarded {i} with non-background pixel ratio: "
                    f"{np.count_nonzero(stack[:, :, i] != 255) / float(stack[:, :, i].size):.3f}",
                    style="warning",
                )
                stack = np.delete(stack, i, axis=-1)

                # Keep PP/PV channel lists consistent with the new stack.
                if self.channels_pp is not None:
                    if i in self.channels_pp:
                        console.print("Channel used for periportal detection removed.", style="warning")
                    self.channels_pp = [j - 1 if j > i else j for j in self.channels_pp if j != i]

                if self.channels_pv is not None:
                    if i in self.channels_pv:
                        console.print("Channel used for perivenous detection removed.", style="warning")
                    self.channels_pv = [j - 1 if j > i else j for j in self.channels_pv if j != i]

            if (self.channels_pp is not None and not self.channels_pp) and (
                    self.channels_pv is not None and not self.channels_pv):
                console.print("No remaining channels for perivenous or periportal detection available.", style="error")
                raise Exception("No remaining channels for segmentation available.")

        return stack, fg_maxproj, bbox, orig_shapes


    def apply(self) -> Tuple[Metadata, Metadata]:

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        new_uid = f"{timestamp}-{short_id}"

        report_path = OUTPUT_PATH / f"segmentation-{new_uid}"
        report_path.mkdir(parents=True, exist_ok=True)

        # Wizard-style pipeline so interactive napari previews can go forward AND backward.
        #
        # Steps:
        #   0 = preprocessing preview (bg_low_val / clahe_bg_suppress / preview_level)
        #   1 = load + filter
        #   2 = weighting preview (nonlinear + alpha)
        #   3 = vessel detection (may raise PipelineBack)
        step = 0

        img_stack: Optional[np.ndarray] = None
        # Flatfield+CLAHE corrected stack, before the blur/p99 that produces
        # img_stack.  Vessel detection reads this one (step 3); blurring first
        # destroys small lumens.
        stack_corr: Optional[np.ndarray] = None
        fg_maxproj: Optional[np.ndarray] = None
        bbox: Optional[Tuple[int, int, int, int]] = None
        orig_shapes: Optional[Dict[int, Any]] = None
        img_size_base: Optional[Tuple[int, int]] = None

        # Interactive stages are wrapped in one loop so each preview's Back
        # button has somewhere to return to. Without it the seeding and
        # smoothing previews sit outside the scope that handles PipelineBack,
        # which is why Back was hidden there.
        #   _entry_step:   where the step machine (re)starts; 3 = vessel detection
        #   _redo_seeding: skip the step machine and redo seeding only
        _entry_step = 0
        _redo_seeding = False
        _back_to_seeding = False
        while True:
            if _redo_seeding:
                _redo_seeding = False
            else:
                step = _entry_step
            while True:
                if step == 0:
                    # Initial interactive preprocessing preview (napari). The
                    # preview_level selector uses the intersection across stains,
                    # parameters are staged on change and updates apply only on
                    # "Preview / Update", and confirm/back is handled in the napari
                    # UI when preview is enabled.
                    if self.preview:
                        action = self._interactive_preprocessing_preview(
                            report_path=report_path,
                            require_confirm=self.confirm,
                            return_action=True,
                        )
                        if action == "abort":
                            console.print("Aborted by user. No segmentation performed.", style="error")
                            return self.metadata
                        if action == "back":
                            console.print("Aborted by user. No segmentation performed.", style="error")
                            return self.metadata
                        if self.confirm and action != "confirm":
                            console.print("Aborted by user. No segmentation performed.", style="error")
                            return self.metadata

                    step = 1
                    continue

                if step == 1:
                    console.print("Loading images...", style="info")
                    img_stack, fg_maxproj, bbox, orig_shapes = self._load_and_invert_images_from_metadatas(report_path)

                    img_size_base = img_stack.shape[:2]  # base size for back-mapping of mask

                    console.print("Complete. Now applying filters...", style="info")
                    # Keep both halves of preprocessing.  Vessel detection needs the
                    # corrected-but-unblurred stack (blurring destroys small lumens);
                    # the PV/PP stains need the blurred, p99-normalised one.
                    stack_corr = self._correct_stack(img_stack)
                    img_stack = self._finalize_stains(stack_corr)

                    for i in range(img_stack.shape[2]):
                        cv2.imwrite(str(report_path / f"slide_{i}.png"), img_stack[:, :, i])

                    if (not self.preview) and self.confirm:
                        apply = Confirm.ask("Continue with lobule segmentation?", default=True, console=console)
                        if not apply:
                            console.print("Aborted by user. No segmentation performed.", style="error")
                            return self.metadata

                    step = 3
                    continue

                if step == 3:
                    # Vessel detection (steps 3-7)
                    pad = 10  # adjust if you want a different safety border


                    try:
                        # Pixel-level zone classification from the shared module.
                        # Runs on the corrected
                        # stack (flatfield + CLAHE only), padded here because the
                        # shared function does no padding of its own and everything
                        # downstream expects padded coordinates.
                        _u8 = np.clip(stack_corr, 0, 255).astype(np.uint8)
                        _u8 = np.pad(_u8, ((pad, pad), (pad, pad), (0, 0)),
                                     mode="constant", constant_values=0)

                        def _run_zone_detector():
                            """Re-runnable so the preview below can re-detect on change."""
                            return detect_and_classify_vessels(
                                _u8,
                                has_pp=self._has_pp_channel(_u8.shape[-1]),
                                pv_index=self._pv_index(),
                                pp_index=self._pp_index(),
                                ksize=self.ksize,
                                vessel_annulus_px=self.vessel_annulus_px,
                                vessel_circularity_min=self.vessel_circularity_min,
                                vessel_max_area_px=self.vessel_max_area_px,
                                vessel_blur_px=self.vessel_blur_px,
                                min_vessel_area_pp=self.min_vessel_area_pp,
                                min_vessel_area_pv=self.min_vessel_area_pv,
                                vessel_tophat_frac=self.vessel_tophat_frac,
                                vessel_tophat_kernel_px=self.vessel_tophat_kernel_px,
                                vessel_pv_zone_thr=self.vessel_pv_zone_thr,
                                vessel_pp_zone_thr=self.vessel_pp_zone_thr,
                                vessel_zone_sigma=self.vessel_zone_sigma,
                                vessel_min_enclosure=self.vessel_min_enclosure,
                                stain_indices=self._stain_indices(_u8.shape[-1]),
                                vessel_lumen_pct=self.vessel_lumen_pct,
                                vessel_lumen_blur_px=self.vessel_lumen_blur_px,
                            )

                        cv_mask_full, pf_mask, tissue_mask, vessel_holes = _run_zone_detector()

                        # Portality and the watershed do not need the same strictness,
                        # and they already receive separate masks. The enclosure gate
                        # that keeps the watershed's foreground clean also rejects most
                        # genuine lumens, starving lobule_portality of the anchors that
                        # place its 1.0 end. Measured on the annotated slide: a looser
                        # pass lifts portality anchors from 2/60 to 44/60 while the
                        # watershed, still fed the strict masks, moves 0.0 SE.
                        _cv_portality, _pf_portality = cv_mask_full, pf_mask
                        if self.vessel_portality_enclosure is not None:
                            _loose = detect_and_classify_vessels(
                                _u8,
                                has_pp=self._has_pp_channel(_u8.shape[-1]),
                                pv_index=self._pv_index(), pp_index=self._pp_index(),
                                ksize=self.ksize,
                                vessel_annulus_px=self.vessel_annulus_px,
                                vessel_circularity_min=self.vessel_circularity_min,
                                vessel_max_area_px=self.vessel_max_area_px,
                                vessel_blur_px=self.vessel_blur_px,
                                min_vessel_area_pp=self.min_vessel_area_pp,
                                min_vessel_area_pv=self.min_vessel_area_pv,
                                vessel_tophat_frac=self.vessel_tophat_frac,
                                vessel_tophat_kernel_px=self.vessel_tophat_kernel_px,
                                vessel_pv_zone_thr=self.vessel_pv_zone_thr,
                                vessel_pp_zone_thr=self.vessel_pp_zone_thr,
                                vessel_zone_sigma=self.vessel_zone_sigma,
                                vessel_min_enclosure=self.vessel_portality_enclosure,
                                stain_indices=self._stain_indices(_u8.shape[-1]),
                                vessel_lumen_pct=self.vessel_lumen_pct,
                                vessel_lumen_blur_px=self.vessel_lumen_blur_px,
                            )
                            _cv_portality, _pf_portality = _loose[0], _loose[1]
                            console.print(
                                f"Portality vessels (enclosure "
                                f"{self.vessel_portality_enclosure:.2f}): "
                                f"{int(_cv_portality.sum())} CV px, "
                                f"{int(_pf_portality.sum())} portal px", style="info")

                        if self.preview:
                            cv_mask_full, pf_mask, tissue_mask, vessel_holes = \
                                self._preview_zone_vessels(_u8, _run_zone_detector)
                        vessel_classes, vessel_contours = _masks_to_contours(
                            cv_mask_full, pf_mask)
                        console.print(
                            f"Vessels: {int((np.array(vessel_classes) == 0).sum())} central, "
                            f"{int((np.array(vessel_classes) == 1).sum())} portal "
                            f"({int(cv_mask_full.sum())} / {int(pf_mask.sum())} px)",
                            style="info",
                        )
                        if report_path is not None:
                            _rep = np.zeros(cv_mask_full.shape + (4,), dtype=np.uint8)
                            _rep[cv_mask_full] = (255, 255, 0, 127)
                            _rep[pf_mask] = (255, 0, 255, 127)
                            cv2.imwrite(str(report_path / "classified_vessels.png"), _rep)

                    except PipelineAbort:
                        console.print("Aborted by user. No segmentation performed.", style="error")
                        return self.metadata

                    except PipelineBack as e:
                        # All Back buttons in the vessel/final previews map here.
                        target = int(e.target_step)

                        step_name = {
                            0: "preprocessing preview",
                            1: "load + filter",
                            3: "vessel detection",
                            4: "post-processing + final preview",
                        }.get(target, f"step {target}")

                        console.print(f"Moving back to {step_name}...", style="warning")

                        step = target
                        continue

                    # Successful completion of the interactive portion.
                    break

            console.print("Complete. Running Higra seeded watershed...", style="info")

            from slidekick.processing.lobule_segmentation.higra_segment import (
                segment_lobules,
                _masked_smooth as _higra_masked_smooth,
                _smooth_labels_valley_aware as _higra_smooth_labels,
            )

            _pv_raw = self._combine_channels(img_stack, self.channels_pv)
            pv_stain = np.pad(_pv_raw, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)

            # Build CV mask in padded coordinates for watershed seeding (class 0 = central vein).
            cv_mask = cv_mask_full

            # Decouple seeding from detection: the full cv_mask stays available for
            # portality and the overlays, while small noisy components are dropped
            # from the mask that seeds the watershed.
            cv_mask_seed = filter_cv_seeds(cv_mask, self.min_cv_seed_area)

            # Build PP stain for watershed boundary evidence (applied to pp_stain
            # channel so segment_lobules can reinforce lobule walls at portal tracts).
            if self.channels_pp:
                _pp_raw = self._combine_channels(img_stack, self.channels_pp)
                pp_stain = np.pad(_pp_raw, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
                if self.pp_close_radius > 0:
                    _pp_se = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * self.pp_close_radius + 1, 2 * self.pp_close_radius + 1),
                    )
                    pp_stain = cv2.morphologyEx(pp_stain, cv2.MORPH_CLOSE, _pp_se)
            else:
                pp_stain = None

            # -- Interactive Higra-seeding preview (downsampled) -------------------
            # Shows a fast napari preview so the user can tune blob_sigma,
            # peak_min_dist, peak_thresh_pct, valley_sigma, and min_area_px before
            # the (slow) full-resolution watershed.  Widgets show full-res pixel
            # values; scaling to preview resolution is handled internally.
            if self.preview:
                _PREV_MAX = self.higra_preview_max_px
                Hpad, Wpad = tissue_mask.shape  # already padded by vessel detection
                _s = min(1.0, _PREV_MAX / max(Hpad, Wpad))

                if _s < 1.0:
                    _Hp = max(1, int(round(Hpad * _s)))
                    _Wp = max(1, int(round(Wpad * _s)))
                    _pad_s = max(0, int(round(pad * _s)))
                    _tm_s = cv2.resize(
                        tissue_mask.astype(np.uint8), (_Wp, _Hp),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    _vh_s = cv2.resize(
                        vessel_holes.astype(np.uint8), (_Wp, _Hp),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    _pv_s = cv2.resize(pv_stain, (_Wp, _Hp), interpolation=cv2.INTER_AREA)
                    _cv_s = cv2.resize(
                        cv_mask_seed.astype(np.uint8), (_Wp, _Hp),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    _pp_s = (
                        cv2.resize(pp_stain, (_Wp, _Hp), interpolation=cv2.INTER_AREA)
                        if pp_stain is not None else None
                    )
                else:
                    _Hp, _Wp, _pad_s, _s = Hpad, Wpad, pad, 1.0
                    _tm_s, _vh_s, _pv_s = tissue_mask, vessel_holes, pv_stain
                    _cv_s, _pp_s = cv_mask_seed, pp_stain

                # Downsample the original (unpadded) img_stack for the overlay background.
                # Derive target size from (_Hp, _Wp, _pad_s), not by re-rounding
                # (H*_s), so _img_s matches the _seg_crop shape pixel-exactly.
                _Hv = max(1, _Hp - 2 * _pad_s)
                _Wv = max(1, _Wp - 2 * _pad_s)
                _C0 = img_stack.shape[2]
                _img_s = np.stack(
                    [cv2.resize(img_stack[:, :, c], (_Wv, _Hv), interpolation=cv2.INTER_AREA)
                     for c in range(_C0)],
                    axis=-1,
                )

                def _scale_higra_param(key: str, val: float) -> float:
                    """Scale a full-resolution param value to the preview resolution."""
                    if key in ("blob_sigma", "valley_sigma"):
                        return max(1.0, val * _s)
                    if key == "peak_min_dist":
                        return max(1, int(round(val * _s)))
                    if key == "min_area_px":
                        return max(1, int(round(val * _s * _s)))
                    if key in ("contour_smooth_sigma", "min_seed_border_dist",
                               "valley_merge_sigma", "edge_local_contrast_sigma"):
                        return max(0.0, val * _s)
                    if key == "seed_tissue_ring":
                        return max(1, int(round(val * _s)))
                    # dimensionless: peak_thresh_pct, peak_thresh_factor, edge_grad_blend,
                    # valley_merge_thr_pct, fused_merge_pv_pct, pp_blend, pp_thr
                    return val

                _higra_defaults = dict(
                    blob_sigma=self.blob_sigma,
                    peak_min_dist=self.peak_min_dist,
                    peak_thresh_pct=self.peak_thresh_pct,
                    valley_sigma=self.valley_sigma,
                    min_area_px=self.min_area_px,
                    # Added because these carry the measured effect: the valley-merge
                    # threshold dominates parameter importance, and the PP blend is
                    # worth several points of real-slide IoU.
                    valley_merge_thr_pct=(self.valley_merge_thr_pct
                                          if self.valley_merge_thr_pct is not None else 35.0),
                    pp_blend=self.pp_blend,
                    pp_thr=self.pp_thr if self.pp_thr is not None else 50.0,
                    contour_smooth_sigma=self.contour_smooth_sigma,
                    edge_grad_blend=self.edge_grad_blend,
                    valley_merge_sigma=self.valley_merge_sigma,
                )
                _higra_pending = dict(_higra_defaults)

                # Seeds the user has edited, in PREVIEW coordinates. None means "use
                # whatever the detectors find"; an array means the points layer was
                # edited and those points replace the automatic seeds entirely. The
                # layer is pre-filled with the detected seeds, so an untouched layer
                # round-trips to the same segmentation.
                _seed_override = {"xy": None}
                _seed_found = {"cv": np.empty((0, 2)), "peak": np.empty((0, 2))}

                def _compute_higra_preview_vis() -> np.ndarray:
                    _seg, _dbg = segment_lobules(
                        _tm_s, _vh_s, _pv_s,
                        cv_mask=_cv_s,
                        pp_stain=_pp_s,
                        blob_sigma=_scale_higra_param("blob_sigma", _higra_pending["blob_sigma"]),
                        peak_min_dist=_scale_higra_param("peak_min_dist", _higra_pending["peak_min_dist"]),
                        peak_thresh_pct=float(_higra_pending["peak_thresh_pct"]),
                        valley_sigma=_scale_higra_param("valley_sigma", _higra_pending["valley_sigma"]),
                        cluster_merge_sigma=self.cluster_merge_sigma,
                        valley_merge_thr_pct=float(_higra_pending["valley_merge_thr_pct"]),
                        pp_thr=float(_higra_pending["pp_thr"]),
                        pp_blend=float(_higra_pending["pp_blend"]),
                        min_area_px=_scale_higra_param("min_area_px", _higra_pending["min_area_px"]),
                        contour_smooth_sigma=_scale_higra_param(
                            "contour_smooth_sigma", _higra_pending["contour_smooth_sigma"]),
                        min_seed_border_dist=int(_scale_higra_param(
                            "min_seed_border_dist", self.min_seed_border_dist)),
                        peak_thresh_factor=self.peak_thresh_factor,
                        edge_grad_blend=float(_higra_pending["edge_grad_blend"]),
                        edge_local_contrast_sigma=_scale_higra_param(
                            "edge_local_contrast_sigma", self.edge_local_contrast_sigma),
                        fused_merge_pv_pct=self.fused_merge_pv_pct,
                        seed_tissue_ring=_scale_higra_param(
                            "seed_tissue_ring", self.seed_tissue_ring),
                        valley_merge_sigma=_scale_higra_param(
                            "valley_merge_sigma", _higra_pending["valley_merge_sigma"]),
                        boundary_smooth_sigma=0,
                        seed_points_xy=_seed_override["xy"],
                        return_debug=True,
                    )
                    _seed_found["cv"] = np.asarray(_dbg["cv_seed_xy"]).reshape(-1, 2)
                    _seed_found["peak"] = np.asarray(_dbg["peak_seed_xy"]).reshape(-1, 2)
                    _seg_crop = _seg[_pad_s: _Hp - _pad_s, _pad_s: _Wp - _pad_s] if _pad_s > 0 else _seg
                    return overlay_mask(_img_s, _seg_crop, alpha=0.5)

                def _seed_points_for_layer() -> np.ndarray:
                    """Detected seeds as napari (row, col) in the cropped preview frame.

                    segment_lobules works in PADDED coordinates and reports (x, y);
                    napari wants (row, col) in the cropped frame, so the pad offset
                    comes off here and the axes swap.
                    """
                    pts = np.vstack([_seed_found["cv"], _seed_found["peak"]])                     if len(_seed_found["peak"]) else _seed_found["cv"]
                    pts = np.asarray(pts).reshape(-1, 2)
                    if not len(pts):
                        return np.empty((0, 2))
                    rc = np.column_stack([pts[:, 1] - _pad_s, pts[:, 0] - _pad_s])
                    h, w = _Hp - 2 * _pad_s, _Wp - 2 * _pad_s
                    keep = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
                    return rc[keep]

                def _layer_points_to_seeds(rc) -> np.ndarray:
                    """Inverse of _seed_points_for_layer: napari (row,col) -> padded (x,y)."""
                    rc = np.asarray(rc, float).reshape(-1, 2)
                    if not len(rc):
                        return np.empty((0, 2))
                    return np.column_stack([rc[:, 1] + _pad_s, rc[:, 0] + _pad_s])

                import napari
                from magicgui import magicgui
                _higra_viewer = napari.Viewer()
                _higra_seg_layer = _higra_viewer.add_image(
                    _compute_higra_preview_vis(),
                    name="Higra seeds preview",
                    rgb=True,
                )
                # Editable seed layer, pre-filled with what the detectors found.
                # Add, move or delete points and press "Use edited seeds"; the next
                # recompute uses exactly the layer contents. One marker per point.
                _higra_seed_layer = _higra_viewer.add_points(
                    _seed_points_for_layer(),
                    name="seeds (editable)",
                    size=max(6, int(round(0.012 * max(_Hp, _Wp)))),
                    face_color="#00e5ff",
                    border_color="#003844",
                    symbol="o",
                )

                # These bound values are already rescaled by lobule_diameter_px or
                # um_per_px, so a fixed literal can sit below the current value at
                # another scale, see _widget_range. At the calibration scale these
                # equal the old literals.
                _r_blob = self._widget_range(float(self.blob_sigma), 5.0, 500.0)
                _r_pmd_lo, _r_pmd_hi = (int(v) for v in self._widget_range(
                    float(self.peak_min_dist), 10, 2000))
                _r_valley = self._widget_range(float(self.valley_sigma), 1.0, 200.0)
                _r_ma_lo, _r_ma_hi = (int(v) for v in self._widget_range(
                    float(self.min_area_px), 1000, 500000, grow=16.0))
                # Reference scale for the min-area slider. The value is a full-res
                # area while the preview runs downsampled, so at a coarse base_level
                # the effective threshold can sit orders of magnitude below a real
                # lobule, which is why the slider can look inert. Stating the
                # reference makes it visible.
                _lob_d = float(CALIBRATION_LOBULE_DIAMETER_PX)
                if self.lobule_diameter_px:
                    _lob_d = float(self.lobule_diameter_px)
                elif self.um_per_px:
                    _lob_d = TYPICAL_LOBULE_DIAMETER_UM / float(self.um_per_px)
                _lob_area = np.pi * (_lob_d / 2.0) ** 2
                _min_area_tip = (
                    "Full-resolution minimum lobule area; smaller regions are discarded."
                    "\n"
                    f"REFERENCE: one lobule is about {_lob_d:.0f} px across, "
                    f"= {_lob_area:,.0f} px2 at full resolution.\n"
                    f"The current value is "
                    f"{100.0 * float(self.min_area_px) / max(_lob_area, 1.0):.1f}% of that."
                    "\n"
                    f"This preview runs at {_s:.3f}x, so it acts as "
                    f"{max(1, int(round(float(self.min_area_px) * _s * _s))):,} px2 on screen "
                    f"(areas scale with the square of the preview factor).")
                _r_contour = self._widget_range(float(self.contour_smooth_sigma), 0.0, 30.0)
                _r_vms = self._widget_range(
                    float(_higra_defaults["valley_merge_sigma"]), 0.0, 40.0)

                @magicgui(
                    layout="vertical",
                    auto_call=True,
                    blob_sigma={
                        "min": _r_blob[0], "max": _r_blob[1], "step": 5.0,
                        "label": "Blob sigma (px, full-res)",
                        "tooltip": "Gaussian sigma for PV peak detection. Approximately the lobule radius in full-resolution pixels.",
                    },
                    peak_min_dist={
                        "min": _r_pmd_lo, "max": _r_pmd_hi, "step": 10,
                        "label": "Peak min dist (px, full-res)",
                        "tooltip": "Minimum distance between seed peaks in full-resolution pixels.",
                    },
                    peak_thresh_pct={
                        "min": 5.0, "max": 95.0, "step": 5.0,
                        "label": "Peak thresh (%)",
                        "tooltip": "Percentile brightness threshold for interior PV peaks.",
                    },
                    valley_sigma={
                        "min": _r_valley[0], "max": _r_valley[1], "step": 2.5,
                        "label": "Valley sigma (px, full-res)",
                        "tooltip": "Gaussian sigma for watershed edge weights in full-resolution pixels.",
                    },
                    min_area_px={
                        "min": _r_ma_lo, "max": _r_ma_hi, "step": 1000,
                        "label": "Min lobule area (px², full-res)",
                        "tooltip": _min_area_tip,
                    },
                    valley_merge_thr_pct={
                        "min": 5.0, "max": 80.0, "step": 2.5,
                        "label": "Valley-merge threshold (%)",
                        "tooltip": "Merge adjacent lobules whose shared seam has no PV valley "
                                   "below this percentile. The most influential parameter in tuning.",
                    },
                    pp_blend={
                        "min": 0.0, "max": 1.0, "step": 0.1,
                        "label": "PP blend",
                        "tooltip": "Weight of the periportal stain in the watershed edge map. "
                                   "0 uses PV only; tuning favours 0.6-0.9 where PP exists.",
                    },
                    pp_thr={
                        "min": 0.0, "max": 255.0, "step": 5.0,
                        "label": "PP threshold",
                        "tooltip": "Intensity above which PP counts as a portal tract.",
                    },
                    contour_smooth_sigma={
                        "min": _r_contour[0], "max": _r_contour[1], "step": 1.0,
                        "label": "Contour smoothing (px, full-res)",
                        "tooltip": "Smooths lobule contours after the watershed. 0 disables it.",
                    },
                    edge_grad_blend={
                        "min": 0.0, "max": 1.0, "step": 0.1,
                        "label": "Edge gradient blend",
                        "tooltip": "Weight of the gradient-magnitude term in the edge map.",
                    },
                    use_edited_seeds={
                        "widget_type": "CheckBox",
                        "label": "Use edited seeds",
                        "tooltip": "Take the seeds from the 'seeds (editable)' points "
                                   "layer instead of re-detecting them. Tick this after "
                                   "adding, moving or deleting points; untick to go back "
                                   "to automatic seeding.",
                    },
                    valley_merge_sigma={
                        "min": _r_vms[0], "max": _r_vms[1], "step": 2.0,
                        "label": "Valley-merge sigma (px, full-res)",
                        "tooltip": "Smoothing of the valley field used by the merge test.",
                    },
                )
                def higra_controls(
                    blob_sigma: float = float(self.blob_sigma),
                    peak_min_dist: int = int(self.peak_min_dist),
                    peak_thresh_pct: float = float(self.peak_thresh_pct),
                    valley_sigma: float = float(self.valley_sigma),
                    min_area_px: int = int(self.min_area_px),
                    valley_merge_thr_pct: float = float(_higra_defaults["valley_merge_thr_pct"]),
                    pp_blend: float = float(_higra_defaults["pp_blend"]),
                    pp_thr: float = float(_higra_defaults["pp_thr"]),
                    contour_smooth_sigma: float = float(_higra_defaults["contour_smooth_sigma"]),
                    edge_grad_blend: float = float(_higra_defaults["edge_grad_blend"]),
                    valley_merge_sigma: float = float(_higra_defaults["valley_merge_sigma"]),
                    use_edited_seeds: bool = False,
                ):
                    _higra_pending["blob_sigma"] = float(blob_sigma)
                    _higra_pending["peak_min_dist"] = int(peak_min_dist)
                    _higra_pending["peak_thresh_pct"] = float(peak_thresh_pct)
                    _higra_pending["valley_sigma"] = float(valley_sigma)
                    _higra_pending["min_area_px"] = int(min_area_px)
                    _higra_pending["valley_merge_thr_pct"] = float(valley_merge_thr_pct)
                    _higra_pending["pp_blend"] = float(pp_blend)
                    _higra_pending["pp_thr"] = float(pp_thr)
                    _higra_pending["contour_smooth_sigma"] = float(contour_smooth_sigma)
                    _higra_pending["edge_grad_blend"] = float(edge_grad_blend)
                    _higra_pending["valley_merge_sigma"] = float(valley_merge_sigma)

                def _apply_higra_pending_to_self() -> None:
                    self.blob_sigma = float(_higra_pending["blob_sigma"])
                    self.peak_min_dist = int(_higra_pending["peak_min_dist"])
                    self.peak_thresh_pct = float(_higra_pending["peak_thresh_pct"])
                    self.valley_sigma = float(_higra_pending["valley_sigma"])
                    self.min_area_px = int(_higra_pending["min_area_px"])
                    self.valley_merge_thr_pct = float(_higra_pending["valley_merge_thr_pct"])
                    self.pp_blend = float(_higra_pending["pp_blend"])
                    self.pp_thr = float(_higra_pending["pp_thr"])
                    self.contour_smooth_sigma = float(_higra_pending["contour_smooth_sigma"])
                    self.edge_grad_blend = float(_higra_pending["edge_grad_blend"])
                    self.valley_merge_sigma = float(_higra_pending["valley_merge_sigma"])
                    # Seeds are in PREVIEW pixels; the full-resolution run needs them
                    # divided by the preview scale. Stored on the instance the same way
                    # the tuned parameters are, so apply() picks them up below.
                    if _seed_override["xy"] is not None and len(_seed_override["xy"]):
                        self._edited_seed_points_xy = (
                            np.asarray(_seed_override["xy"], float) / max(_s, 1e-9))
                        console.print(
                            f"Using {len(self._edited_seed_points_xy)} manually edited "
                            f"seed(s) for the full-resolution segmentation.", style="info")
                    else:
                        self._edited_seed_points_xy = None

                def on_higra_update() -> None:
                    console.print(
                        f"Higra preview: blob_sigma={_higra_pending['blob_sigma']:.0f}, "
                        f"peak_min_dist={_higra_pending['peak_min_dist']}, "
                        f"valley_sigma={_higra_pending['valley_sigma']:.1f}, "
                        f"min_area_px={_higra_pending['min_area_px']}",
                        style="info",
                    )
                    _seed_override["xy"] = (
                        _layer_points_to_seeds(_higra_seed_layer.data)
                        if bool(_higra_pending.get("use_edited_seeds")) else None)
                    _higra_seg_layer.data = _compute_higra_preview_vis()
                    _refresh_seed_layer()

                def _refresh_seed_layer() -> None:
                    """Show the seeds the last recompute actually used.

                    While seeding is automatic the layer follows the detectors, so
                    changing a parameter shows its effect on the seeds immediately.
                    Once the user has taken manual control the layer is left alone -
                    otherwise a slider nudge would silently discard their edits.
                    """
                    if _seed_override["xy"] is None:
                        _higra_seed_layer.data = _seed_points_for_layer()

                def on_higra_reset() -> None:
                    _higra_pending.update(_higra_defaults)
                    higra_controls.blob_sigma.value = _higra_defaults["blob_sigma"]
                    higra_controls.peak_min_dist.value = _higra_defaults["peak_min_dist"]
                    higra_controls.peak_thresh_pct.value = _higra_defaults["peak_thresh_pct"]
                    higra_controls.valley_sigma.value = _higra_defaults["valley_sigma"]
                    higra_controls.min_area_px.value = _higra_defaults["min_area_px"]
                    for _k in ("valley_merge_thr_pct", "pp_blend", "pp_thr",
                               "contour_smooth_sigma", "edge_grad_blend", "valley_merge_sigma"):
                        getattr(higra_controls, _k).value = _higra_defaults[_k]
                    _seed_override["xy"] = None          # back to automatic seeding
                    _higra_seg_layer.data = _compute_higra_preview_vis()
                    _higra_seed_layer.data = _seed_points_for_layer()

                _higra_action = run_napari_preview_action(
                    _higra_viewer,
                    higra_controls,
                    require_confirm=False,
                    on_update=on_higra_update,
                    on_reset=on_higra_reset,
                    include_back=True,
                    back_text="Back to vessel detection",
                )

                if _higra_action == "abort":
                    console.print("Aborted by user at Higra seeding step.", style="error")
                    return self.metadata

                if _higra_action == "back":
                    # Seeds follow from the CV mask, so retuning seeding alone
                    # cannot fix a bad vessel detection, so go back to step 3.
                    console.print("Returning to vessel detection...", style="warning")
                    _entry_step = 3
                    continue

                # Commit staged widget values to self regardless of confirm vs apply_values.
                _apply_higra_pending_to_self()
                console.print(
                    f"Higra params confirmed: blob_sigma={self.blob_sigma:.0f}, "
                    f"peak_min_dist={self.peak_min_dist}, valley_sigma={self.valley_sigma:.1f}, "
                    f"min_area_px={self.min_area_px}",
                    style="info",
                )


            # -- Full-resolution watershed, input diagnostics ----------------------
            _fg_diag = tissue_mask & ~vessel_holes
            _fg_px   = int(_fg_diag.sum())
            _tm_px   = int(tissue_mask.sum())
            _vh_px   = int(vessel_holes.sum())

            # CV components: detected vs. those large enough to actually seed the
            # watershed (min_cv_seed_area drops the rest).
            from scipy.ndimage import label as _diag_label
            _cv_labeled, _n_cv = _diag_label(cv_mask)
            _cv_px = int(cv_mask.sum())
            _, _n_cv_seed = _diag_label(cv_mask_seed)
            if _n_cv_seed != _n_cv:
                console.print(
                    f"CV components: {_n_cv} detected, {_n_cv_seed} used as seeds "
                    f"(min_cv_seed_area={self.min_cv_seed_area})",
                    style="info",
                )

            # PV stain range inside tissue
            _pv_fg = pv_stain[tissue_mask]
            _pv_min  = float(_pv_fg.min())  if _pv_fg.size else 0.0
            _pv_max  = float(_pv_fg.max())  if _pv_fg.size else 0.0
            _pv_p50  = float(np.percentile(_pv_fg, 50)) if _pv_fg.size else 0.0
            _pv_p95  = float(np.percentile(_pv_fg, 95)) if _pv_fg.size else 0.0

            # PP stain range inside tissue (if available)
            if pp_stain is not None:
                _pp_fg  = pp_stain[tissue_mask]
                _pp_str = (
                    f"  pp_stain (tissue): min={_pp_fg.min():.1f}  "
                    f"p50={np.percentile(_pp_fg, 50):.1f}  "
                    f"p95={np.percentile(_pp_fg, 95):.1f}  "
                    f"max={_pp_fg.max():.1f}"
                    + (f"  [close_r={self.pp_close_radius}]" if self.pp_close_radius > 0 else "")
                )
            else:
                _pp_str = "  pp_stain: None (no PP channel)"

            console.print("Higra watershed inputs:", style="info")
            console.print(
                f"  tissue_mask:  {_tm_px:,} px  |  vessel_holes: {_vh_px:,} px  |  fg (tissue−holes): {_fg_px:,} px",
                style="info",
            )
            console.print(
                f"  CV seeds:     {_n_cv} component(s)  ({_cv_px:,} px, "
                + ("detected by vessel classifier" if _n_cv > 0 else "none detected → intensity-peak seeding only") + ")",
                style="info",
            )
            console.print(
                f"  pv_stain (tissue): min={_pv_min:.1f}  p50={_pv_p50:.1f}  p95={_pv_p95:.1f}  max={_pv_max:.1f}",
                style="info",
            )
            console.print(_pp_str, style="info")
            console.print(
                f"  params: blob_sigma={self.blob_sigma:.0f}  peak_min_dist={self.peak_min_dist}  "
                f"peak_thresh_pct={self.peak_thresh_pct:.1f}  valley_sigma={self.valley_sigma:.1f}  "
                f"cluster_merge_sigma={self.cluster_merge_sigma:.1f}  min_area_px={self.min_area_px:,}",
                style="info",
            )

            # -- Save Higra input debug images -------------------------------------
            import matplotlib
            matplotlib.use("Agg")

            _THUMB_PX = 900  # max edge (px) for overview thumbnails

            def _to_u8_higra(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
                """Normalise float32 to uint8 using the tissue mask as the reference range."""
                a = arr.astype(np.float32)
                vals = a[mask.astype(bool)] if mask is not None else a[a > 0]
                if vals.size == 0 or float(vals.max()) <= float(vals.min()):
                    return np.zeros(arr.shape, dtype=np.uint8)
                lo, hi = float(vals.min()), float(vals.max())
                return ((a - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)

            def _thumb_higra(img: np.ndarray) -> np.ndarray:
                h, w = img.shape[:2]
                s = _THUMB_PX / max(h, w)
                if s >= 1.0:
                    return img
                nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
                interp = cv2.INTER_NEAREST if img.dtype == bool or img.dtype == np.uint8 and img.ndim == 2 else cv2.INTER_AREA
                return cv2.resize(img.astype(np.uint8) if img.dtype == bool else img, (nw, nh), interpolation=interp)

            # PV maps passed to / derived by segment_lobules
            _pv_blob   = _higra_masked_smooth(pv_stain, tissue_mask, self.blob_sigma)
            _pv_valley = _higra_masked_smooth(pv_stain, tissue_mask, self.valley_sigma)

            # Portal-field mask (class 1) for visualisation alongside CV seeds
            _pf_vis = np.zeros(tissue_mask.shape, dtype=np.uint8)
            _pf_cnts = [c for c, k in zip(vessel_contours, vessel_classes) if k == 1]
            if _pf_cnts:
                cv2.drawContours(_pf_vis, _pf_cnts, -1, 1, cv2.FILLED)
            _pf_mask_vis = _pf_vis.astype(bool)

            # CV-seeds overlay on PV stain (BGR: orange=portal, cyan=CV)
            _pv_u8_vis = _to_u8_higra(pv_stain, tissue_mask)
            _ov_cv = np.stack([_pv_u8_vis, _pv_u8_vis, _pv_u8_vis], axis=-1)
            _ov_cv[_pf_mask_vis] = [0, 140, 255]    # BGR orange = portal field
            _ov_cv[cv_mask]      = [220, 220,   0]  # BGR cyan   = central vein
            cv2.imwrite(str(report_path / "higra_cv_seeds.png"), _ov_cv)

            # Individual grayscale maps
            cv2.imwrite(str(report_path / "higra_pv_stain.png"),
                        _to_u8_higra(pv_stain, tissue_mask))
            cv2.imwrite(str(report_path / "higra_pv_blob.png"),
                        _to_u8_higra(_pv_blob, tissue_mask))
            cv2.imwrite(str(report_path / "higra_pv_valley.png"),
                        _to_u8_higra(_pv_valley, tissue_mask))
            if pp_stain is not None:
                cv2.imwrite(str(report_path / "higra_pp_stain.png"),
                            _to_u8_higra(pp_stain, tissue_mask))

            # Tissue / holes composite (BGR: grey=tissue, red=holes)
            _tis_vis = np.zeros((*tissue_mask.shape, 3), dtype=np.uint8)
            _tis_vis[tissue_mask]  = [190, 190, 190]
            _tis_vis[vessel_holes] = [0,   0,   220]
            cv2.imwrite(str(report_path / "higra_tissue_holes.png"), _tis_vis)

            # Composite overview (matplotlib, RGB panels)
            _ov_panels = [
                (_pv_u8_vis,                          "gray", "PV stain (raw)"),
                (_to_u8_higra(_pv_blob,   tissue_mask), "gray", f"PV blob  σ={self.blob_sigma:.0f} px"),
                (_to_u8_higra(_pv_valley, tissue_mask), "gray", f"PV valley σ={self.valley_sigma:.1f} px"),
                (_ov_cv[:, :, ::-1],                  None,   f"CV seeds (cyan) · portal (orange)  n={_n_cv}"),
                (_tis_vis[:, :, ::-1],                None,   f"Tissue (grey) · holes (red)  {_tm_px:,} / {_vh_px:,} px"),
            ]
            if pp_stain is not None:
                _ov_panels.append((
                    _to_u8_higra(pp_stain, tissue_mask), "gray",
                    f"PP stain (close_r={self.pp_close_radius})",
                ))
            _ov_n     = len(_ov_panels)
            _ov_ncols = 3
            _ov_nrows = (_ov_n + _ov_ncols - 1) // _ov_ncols
            _ov_fig, _ov_axes = plt.subplots(
                _ov_nrows, _ov_ncols, figsize=(_ov_ncols * 4, _ov_nrows * 4),
            )
            _ov_axes = np.array(_ov_axes).ravel()
            for _oi, (_img, _cmap, _title) in enumerate(_ov_panels):
                _ov_axes[_oi].imshow(_thumb_higra(_img), cmap=_cmap, interpolation="nearest")
                _ov_axes[_oi].set_title(_title, fontsize=8)
                _ov_axes[_oi].axis("off")
            for _ax in _ov_axes[_ov_n:]:
                _ax.axis("off")
            _ov_fig.suptitle(
                f"Higra watershed inputs  |  CV={_n_cv} component(s) / {_cv_px:,} px",
                fontsize=9,
            )
            _ov_fig.tight_layout()
            _ov_fig.savefig(
                str(report_path / "higra_inputs_overview.png"), dpi=120, bbox_inches="tight",
            )
            plt.close(_ov_fig)
            console.print(
                f"  Higra debug images saved → {report_path.name}/higra_*.png",
                style="info",
            )

            # -- Full-resolution watershed -----------------------------------------
            # Run watershed without smoothing; smoothing is applied interactively below.
            mask_raw = segment_lobules(
                tissue_mask, vessel_holes, pv_stain,
                cv_mask=cv_mask_seed,
                pp_stain=pp_stain,
                blob_sigma=self.blob_sigma,
                peak_min_dist=self.peak_min_dist,
                peak_thresh_pct=self.peak_thresh_pct,
                valley_sigma=self.valley_sigma,
                cluster_merge_sigma=self.cluster_merge_sigma,
                valley_merge_thr_pct=self.valley_merge_thr_pct,
                pp_thr=self.pp_thr,
                pp_blend=self.pp_blend,
                min_area_px=self.min_area_px,
                contour_smooth_sigma=self.contour_smooth_sigma,
                min_seed_border_dist=int(self.min_seed_border_dist),
                peak_thresh_factor=self.peak_thresh_factor,
                edge_grad_blend=self.edge_grad_blend,
                edge_local_contrast_sigma=self.edge_local_contrast_sigma,
                fused_merge_pv_pct=self.fused_merge_pv_pct,
                seed_tissue_ring=self.seed_tissue_ring,
                valley_merge_sigma=self.valley_merge_sigma,
                seed_points_xy=self._edited_seed_points_xy,
                boundary_smooth_sigma=0,  # applied interactively below
            )

            # Reconstruct the inverted PV field (same formula as inside segment_lobules)
            # so the interactive smoothing step can call _higra_smooth_labels directly.
            _fg_ws = tissue_mask & ~vessel_holes
            _smoothed_ws = _higra_masked_smooth(pv_stain, tissue_mask, self.valley_sigma)
            _mx_ws = float(_smoothed_ws[tissue_mask].max()) if tissue_mask.any() else 1.0
            _inverted_ws = (_mx_ws - _smoothed_ws).astype(np.float32)
            _inverted_ws[~tissue_mask] = _mx_ws

            # === Interactive boundary-smoothing + final-review loop ===
            # Runs once normally; re-iterates when the user presses Back from the
            # final review to re-adjust the smoothing parameters.
            _show_smoothing_preview = True

            while True:

                # -------------------------------------------------------------------
                # A. Interactive boundary-smoothing preview (napari)
                # -------------------------------------------------------------------
                if _show_smoothing_preview and self.preview:
                    _smooth_defaults = dict(
                        boundary_smooth_sigma=self.boundary_smooth_sigma,
                        boundary_smooth_valley_pct=self.boundary_smooth_valley_pct,
                    )
                    _smooth_pending = dict(_smooth_defaults)

                    def _apply_smooth_pending_to_self() -> None:
                        self.boundary_smooth_sigma = float(_smooth_pending["boundary_smooth_sigma"])
                        self.boundary_smooth_valley_pct = float(_smooth_pending["boundary_smooth_valley_pct"])

                    def _compute_smooth_overlay_vis() -> np.ndarray:
                        _s = float(_smooth_pending["boundary_smooth_sigma"])
                        _v = float(_smooth_pending["boundary_smooth_valley_pct"])
                        _m = (
                            _higra_smooth_labels(
                                mask_raw, _inverted_ws, _fg_ws,
                                sigma=_s, valley_pct=_v,
                                valley_band_px=self.boundary_smooth_valley_band_px,
                            )
                            if _s > 0 else mask_raw
                        )
                        return overlay_mask(img_stack, _m[pad:-pad, pad:-pad], alpha=0.5)

                    _smooth_viewer = napari.Viewer()
                    _smooth_seg_layer = _smooth_viewer.add_image(
                        _compute_smooth_overlay_vis(),
                        name="Segmentation (smoothed)",
                        rgb=True,
                    )

                    # boundary_smooth_sigma is rescaled by lobule_diameter_px like the
                    # other pixel-valued parameters, see _widget_range.
                    _r_bss = self._widget_range(float(self.boundary_smooth_sigma), 0.0, 80.0)
                    # boundary_smooth_valley_pct is a percentile (0-100), not scaled by
                    # lobule_diameter_px at all. Its floor of 40.0 sat above the shipped
                    # default of 35.0, which raised ValueError building this widget.
                    # Widened to 5.0, matching the floor already used for the sibling
                    # percentile widget (peak_thresh_pct) elsewhere in this file.
                    @magicgui(
                        layout="vertical",
                        auto_call=True,
                        boundary_smooth_sigma={
                            "min": _r_bss[0], "max": _r_bss[1], "step": 1.0,
                            "label": "Smooth sigma (px)  [0 = off]",
                            "tooltip": "Gaussian radius for boundary rounding. Larger = smoother curves. 0 disables smoothing entirely.",
                        },
                        boundary_smooth_valley_pct={
                            "min": 5.0, "max": 99.0, "step": 1.0,
                            "label": "Freeze valley pct",
                            "tooltip": "Top N% of inverted-PV pixels are frozen (real boundaries stay). Higher = more frozen.",
                        },
                    )
                    def smooth_controls(
                        boundary_smooth_sigma: float = float(self.boundary_smooth_sigma),
                        boundary_smooth_valley_pct: float = float(self.boundary_smooth_valley_pct),
                    ):
                        _smooth_pending["boundary_smooth_sigma"] = float(boundary_smooth_sigma)
                        _smooth_pending["boundary_smooth_valley_pct"] = float(boundary_smooth_valley_pct)

                    def on_smooth_update() -> None:
                        console.print(
                            f"Smoothing: sigma={_smooth_pending['boundary_smooth_sigma']:.0f} px, "
                            f"valley_pct={_smooth_pending['boundary_smooth_valley_pct']:.0f}",
                            style="info",
                        )
                        _smooth_seg_layer.data = _compute_smooth_overlay_vis()

                    def on_smooth_reset() -> None:
                        self.boundary_smooth_sigma = _smooth_defaults["boundary_smooth_sigma"]
                        self.boundary_smooth_valley_pct = _smooth_defaults["boundary_smooth_valley_pct"]
                        _smooth_pending.update(_smooth_defaults)
                        smooth_controls.boundary_smooth_sigma.value = _smooth_defaults["boundary_smooth_sigma"]
                        smooth_controls.boundary_smooth_valley_pct.value = _smooth_defaults["boundary_smooth_valley_pct"]
                        _smooth_seg_layer.data = _compute_smooth_overlay_vis()

                    def on_smooth_confirm() -> None:
                        pass  # last Recalculate result is what was previewed; don't recompute

                    def on_smooth_apply_values() -> None:
                        _apply_smooth_pending_to_self()

                    _smooth_action = run_napari_preview_action(
                        _smooth_viewer,
                        smooth_controls,
                        require_confirm=False,
                        on_update=on_smooth_update,
                        on_reset=on_smooth_reset,
                        on_confirm=on_smooth_confirm,
                        on_apply_values=on_smooth_apply_values,
                        include_back=True,
                        back_text="Back to seeding",
                    )

                    if _smooth_action == "abort":
                        console.print("Aborted by user at smoothing step.", style="error")
                        return self.metadata

                    if _smooth_action == "back":
                        console.print("Returning to the seeding preview...",
                                      style="warning")
                        _back_to_seeding = True
                        break

                _show_smoothing_preview = False

                # -------------------------------------------------------------------
                # B. Apply confirmed smoothing parameters
                # -------------------------------------------------------------------
                if self.boundary_smooth_sigma > 0:
                    mask = _higra_smooth_labels(
                        mask_raw, _inverted_ws, _fg_ws,
                        sigma=self.boundary_smooth_sigma,
                        valley_pct=self.boundary_smooth_valley_pct,
                        valley_band_px=self.boundary_smooth_valley_band_px,
                    )
                else:
                    mask = mask_raw

                # Crop mask by padding
                mask_cropped = mask[pad:-pad, pad:-pad]

                # ----------------------------------------------------------------
                # C. Base-level full-frame mask + portality
                # ----------------------------------------------------------------

                # Step 10a: Build base-level full-frame mask only (no pyramid yet).
                proc_h, proc_w = mask_cropped.shape
                Hb, Wb = img_size_base
                Hfull_base, Wfull_base = orig_shapes[self.base_level]
                min_r, min_c, max_r, max_c = bbox

                roi_base = cv2.resize(
                    mask_cropped.astype(np.int32), (Wb, Hb),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
                mask_base_full = np.zeros((Hfull_base, Wfull_base), dtype=np.int32)
                r0 = max(0, min_r); c0 = max(0, min_c)
                r1 = min(Hfull_base, max_r); c1 = min(Wfull_base, max_c)
                if (r1 - r0) != Hb or (c1 - c0) != Wb:
                    roi_base = cv2.resize(roi_base, (c1 - c0, r1 - r0),
                                          interpolation=cv2.INTER_NEAREST).astype(np.int32)
                mask_base_full[r0:r1, c0:c1] = roi_base

                console.print("Complete. Creating portality map...", style="info")

                # Step 10b: Vessel contours in base-level full-frame coordinates.
                # Portality uses the (optionally looser) vessel set; the watershed above
                # has already used the strict one. See vessel_portality_enclosure.
                if self.vessel_portality_enclosure is not None:
                    _pcls, _pcnt = _masks_to_contours(_cv_portality, _pf_portality)
                else:
                    _pcls, _pcnt = vessel_classes, vessel_contours
                cv_cnt_roi = [c for c, k in zip(_pcnt, _pcls) if k == 0]
                pf_cnt_roi = [c for c, k in zip(_pcnt, _pcls) if k == 1]

                cv_cnt_base = to_base_full(cv_cnt_roi, pad, bbox, (proc_h, proc_w), (Hb, Wb))
                pf_cnt_base = to_base_full(pf_cnt_roi, pad, bbox, (proc_h, proc_w), (Hb, Wb))

                # Step 10c: Portality at base_level only.
                _canvas_cv = np.zeros((Hfull_base, Wfull_base), dtype=np.uint8)
                _canvas_pf = np.zeros((Hfull_base, Wfull_base), dtype=np.uint8)
                for c in cv_cnt_base:
                    cv2.fillPoly(_canvas_cv, [c.reshape(-1, 1, 2).astype(np.int32)], 1)
                for c in pf_cnt_base:
                    cv2.fillPoly(_canvas_pf, [c.reshape(-1, 1, 2).astype(np.int32)], 1)

                # Build PV stain in full-frame coordinates for the CV-fallback in
                # lobule_portality (lobules without an annotated CV use the PV
                # intensity peak as a surrogate central vein).
                _pv_roi = cv2.resize(_pv_raw, (c1 - c0, r1 - r0),
                                     interpolation=cv2.INTER_LINEAR)
                _pv_full = np.zeros((Hfull_base, Wfull_base), dtype=np.float32)
                _pv_full[r0:r1, c0:c1] = _pv_roi

                P_base = lobule_portality(
                    mask_base_full,
                    _canvas_cv.astype(bool),
                    _canvas_pf.astype(bool),
                    pv_stain=_pv_full,
                ).astype(np.float32)

                # Crop base-level portality to ROI and save fixed-size PNG
                portality_cropped = P_base[min_r:max_r, min_c:max_c]

                cmap = plt.get_cmap("magma").copy()
                cmap.set_bad(alpha=0.0)
                Pm = np.ma.masked_invalid(portality_cropped).astype(np.float32)

                # Vessel lumens are NaN, like the background, but they mean
                # something different: vessel here, no tissue to zonate, rather
                # than nothing here. Drawn transparent they read as white, and
                # magma's top colour is a pale cream, so a lumen and a portality
                # of 1.0 look the same and a central vein appears to be 1 itself
                # instead of being a hole ringed by 1. Painting them grey keeps
                # the distinction legible.
                _lumen = vessel_lumen_mask(portality_cropped)
                _port_rgba = (cmap(Pm) * 255).astype(np.uint8)
                _port_rgba[_lumen, :3] = LUMEN_DISPLAY_RGB
                _port_rgba[_lumen, 3] = 255

                plt.imsave(str(report_path / "portality.png"), _port_rgba)

                # ----------------------------------------------------------------
                # D. Instance overlay, written for every run, not just interactive
                # ----------------------------------------------------------------
                # This used to be built inside the `if self.preview` block below and
                # only ever shown in napari, so a headless run produced the mask and
                # portality TIFFs but no image you could actually look at.
                overlay_vis = overlay_mask(img_stack, mask_cropped, alpha=0.5)
                cv2.imwrite(str(report_path / "segmentation_overlay.png"),
                            cv2.cvtColor(overlay_vis, cv2.COLOR_RGB2BGR))
                console.print(
                    f"Instance overlay -> {report_path / 'segmentation_overlay.png'} "
                    f"({int(mask_cropped.max())} lobules)", style="info")

                # ----------------------------------------------------------------
                # E. Final review preview
                # ----------------------------------------------------------------
                if self.preview:

                    # Same lumen treatment as the saved PNG above, so the preview
                    # and the report agree on what a grey hole means.
                    portality_rgba = _port_rgba.copy()
                    portality_rgba[..., 3] = (portality_rgba[..., 3] * 0.85).astype(np.uint8)

                    action = preview_images_napari(
                        [overlay_vis, portality_rgba],
                        titles=["Segmentation Overlay", "Portality"],
                        require_confirm=self.confirm,
                        return_action=True,
                        include_confirm=True,
                        include_back=True,
                        confirm_text="Confirm and finish",
                        back_text="Back to smoothing",
                    )

                    if action == "back":
                        # Return to smoothing preview so user can tweak sigma / valley_pct.
                        _show_smoothing_preview = True
                        continue

                    if self.confirm and action == "closed":
                        console.print("Aborted by user at final preview. No outputs written.", style="error")
                        return self.metadata

                break  # confirmed (or no preview)

            # Back at the smoothing preview: redo seeding without re-running
            # vessel detection.
            if _back_to_seeding:
                _back_to_seeding = False
                _redo_seeding = True
                continue
            break

        # Step 11: Build full pyramids only now (for saving).
        console.print("Building pyramids for all levels...", style="info")

        mask_pyramid: Dict[int, np.ndarray] = {}
        portality_pyramid: Dict[int, np.ndarray] = {}
        for lvl, (Hdst, Wdst) in orig_shapes.items():
            if lvl == self.base_level:
                mask_pyramid[lvl] = mask_base_full
                portality_pyramid[lvl] = P_base
            else:
                mask_pyramid[lvl] = cv2.resize(
                    mask_base_full, (Wdst, Hdst),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
                portality_pyramid[lvl] = cv2.resize(
                    P_base, (Wdst, Hdst),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)

        # Save mask pyramid
        seg_path = report_path / f"{new_uid}_seg.tiff"

        new_meta = Metadata(
            path_original=seg_path,
            path_storage=seg_path,
            image_type="mask",
            uid=new_uid + "_mask",
        )
        new_meta.save(report_path)
        save_tif(mask_pyramid, seg_path, metadata=new_meta)

        # Save portality pyramid
        portality_path = report_path / f"{new_uid}_portality.tiff"
        new_port = Metadata(
            path_original=portality_path,
            path_storage=portality_path,
            image_type="portality",
            uid=new_uid + "_portality",
        )
        new_port.save(report_path)
        save_tif(portality_pyramid, portality_path, metadata=new_port)

        console.print("Complete.", style="info")

        # Return both metadata objects
        return new_meta, new_port


if __name__ == "__main__":
    # Example usage. Flip this to choose the input: the stored files on disk, or a
    # freshly generated synthetic sample. Both paths end in the same .apply() call.
    USE_SYNTHETIC = True

    if not USE_SYNTHETIC:
        # stored files
        from slidekick import DATA_PATH

        image_paths = [DATA_PATH / "reg_n_sep" / "noise.tiff",
                       #DATA_PATH / "reg_n_sep" / "periportal.tiff",
                       DATA_PATH / "reg_n_sep" / "perivenous.tiff",
                       ]
        metadata_for_segmentation = [
            Metadata(path_original=p, path_storage=p) for p in image_paths]
        segmentor = LobuleSegmentor(metadata_for_segmentation,
                                    #channels_pp=1,
                                    channels_pv=1,
                                    base_level=0,
                                    adaptive_histonorm=True)
    else:
        # generated synthetic sample
        # Uses the src-level generator directly (importing from examples/ would be
        # a backwards dependency). One instance -> 3 channels: 0 = PV marker,
        # 1 = PP marker, 2 = structural.
        import slidekick.processing.lobule_segmentation.synth_lobules.synthetic_lobules as _synth
        _synth.IMG = 2000
        from slidekick.processing.lobule_segmentation.synth_lobules.synthetic_lobules import (
            generate_all_instances,
        )

        inst = generate_all_instances(
            seeds=(12,), hex_sides=(200,), pv_stain_types=("hill",),
            pp_stain_types=("linear",), noise_levels=("low",), modes=("dual",))[0]
        stack = inst.image_stack

        out_dir = OUTPUT_PATH / "synthetic_example"
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_for_segmentation = []
        for c, name in enumerate(("PV_marker", "PP_marker", "structural")):
            ch = stack[:, :, c]
            p = out_dir / f"synthetic_ch{c}_{name}.tiff"
            save_tif({0: ch, 1: ch[::2, ::2], 2: ch[::4, ::4]}, p)
            metadata_for_segmentation.append(Metadata(path_original=p, path_storage=p))

        # Synthetic lobules at hex_side=200 are ~311 px across, against the 226.4 px
        # the defaults are calibrated for, so the lobule-scale parameters must be
        # scaled up. But the generator keeps vessel and noise size fixed, so the
        # vessel-area and noise-kernel gates must not scale (that would reject the
        # synthetic central veins). Take the scaled defaults, then drop those back
        # to shipped.
        kw = LobuleSegmentor.scaled_defaults(lobule_diameter_px=311.0)
        for k in ("min_vessel_area_pp", "min_vessel_area_pv", "vessel_max_area_px",
                  "min_cv_seed_area", "vessel_annulus_px", "vessel_blur_px",
                  "vessel_tophat_kernel_px", "ksize", "clahe_tile_px"):
            kw.pop(k, None)
        segmentor = LobuleSegmentor(metadata_for_segmentation,
                                    channels_pv=[0], channels_pp=[1],
                                    base_level=0, **kw)

    metadata_segmentation, metadata_portality = segmentor.apply()
