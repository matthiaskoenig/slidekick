"""
Synthetic lobule generator with realistic properties.

Generates connected hexagonal lobules with:
- Oscillating (back-and-forth) curved boundaries via Voronoi + sinusoidal perturbation
- Accurate per-lobule portality from actual boundary geometry
- Portal triads (1-3 vessels) at Voronoi vertices with random jitter
- Central vein ellipses at lobule centroids
- Multiple stain models: linear/sharp × PV/PP × 2 noise levels
- Scan-like background with tissue blob

Usage:
    from synthetic_lobules import generate_all_instances, SyntheticInstance
"""

import json
import math
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.ndimage import label as _ndlabel, binary_dilation as _nddilate, generate_binary_structure as _ndstruct
from scipy.spatial import Voronoi
from skimage.draw import polygon as ski_polygon

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
# Canvas size - chosen to roughly match the SMALLER real multiplexed fluorescence images
# at level 2 (~1800–3000 px). Synthetic pixel scale therefore ≈ real level-2
# pixel scale, so pipeline parameters (min_vessel_area, ridge_width, ...)
# transfer between synthetic and real without rescaling.
IMG = 1600
SCAN_GRAY = 8
SCAN_MARGIN = 20

# Vessel sizes - calibrated to real CV detections at level 2.
# CV aspect ratios are now independent (rx != ry) to simulate oblique sections
# where the vessel appears elongated.  Blob noise is also wider so circularity
# spans a realistic range (0.2–0.6) matching real slides.
CV_RX = (4, 16)               # minor semi-axis
CV_RY = (4, 24)               # major semi-axis - moderate elongation
CV_DROPOUT = 0.15             # 15% of lobules have no central vein
PV_SIGMA_FRAC = 1 / 4         # exponential half-max for PV sharp stain (faster decay)
PP_RX = (2, 10)               # portal vessels: slightly more variable sizes
PP_RY = (3, 14)

# Expression heterogeneity: per-lobule brightness multiplier range
LOBULE_EXPR_MIN = 0.35        # weakest lobule retains 35% of max expression
LOBULE_EXPR_MAX = 1.0

# Global brightness variability: real images differ 2-3x in mean intensity.
# The PP marker channel in particular varies whole-slide mean from 27 -> 97 across the 7
# real slides, so we allow a large multiplier.
GLOBAL_BRIGHTNESS_MIN = 0.35
GLOBAL_BRIGHTNESS_MAX = 1.00

# Autofluorescence: baseline signal even in PP-dark regions. Real PV marker
# has p05 ≈ 15, mean ≈ 76, so the PP baseline is ~20 % of the peak.
AUTOFLUO_FRAC = 0.12

# Tissue folds: bright linear artifacts
FOLD_PROB = 0.0               # probability of having a fold (disabled)
FOLD_INTENSITY = 0.7          # relative brightness of fold

# Master toggles for non-stain image artifacts. Set False to remove
# scanbox / illumination artifacts from the rendered output.
ENABLE_SCANBOX = False
ENABLE_ILLUMINATION = True

# Boundary perturbation - Gaussian-bump curved edges.
# 7 intermediate control points per edge give a smooth organic curve.
# Displacement follows a single Gaussian bump (amplitude, center, width
# sampled randomly per edge), then a Gaussian-1D smoothing pass is applied
# to 90% of edges for continuity ("Stetigkeit"); 10% remain unsmoothed.
PERTURB_N_PTS = 20            # intermediate control points per edge (smoother curves)
PERTURB_ORTHO_FRAC = 0.22     # max orthogonal shift as fraction of edge length (was 0.58)
PERTURB_ALONG_FRAC = 0.04     # max along-edge shift (endpoint shift only)
VERTEX_JITTER_FRAC = 0.06     # max shift of Voronoi corners as fraction of hex side (was 0.10)

# Portal triad: 1-3 vessels per vertex, 30% chance of missing entirely
PP_TRIAD_MAX = 4
PP_DROPOUT = 0.15
PP_JITTER_FRAC = 0.10         # jitter around vertex as fraction of hex side

# Lobule boundary fusion: each Voronoi edge between adjacent lobules gets a
# random value v ~ U[0,1]. If v < FUSION_THRESHOLD the boundary is absent
# from GT (lobules merge into one super-lobule). Portality floor at that
# edge rises proportionally (max floor when v=0, zero floor at threshold).
# Portal vessels that become interior after merging are removed.
FUSION_THRESHOLD   = 0.41              # v < threshold -> merge in GT
#                                        with v_power=3: P(fuse) = 0.41^3 ≈ 6.9% per edge
#                                        4-connected adjacency detects ~94 pairs for 47 lobules
#                                        -> expected ~6-7 fused edges per image
FUSION_V_POWER     = 3.0               # v = U^(1/power): skews toward 1


# ---------------------------------------------------------------------------
#  GT-derived calibration
# ---------------------------------------------------------------------------
# Produced by output/calibrate_synth_from_gt.py from the manually annotated
# real slides. When present, the JSON overrides a small set of module-level
# constants and the defaults of `generate_all_instances` so the synthetic
# stains, noise channel, and lobule geometry match real level-2 data.
#
# Numbers that came out of the fit (for reference, not hard-coded here):
#   PV (PV marker):  base ≈ 0.163, k ≈ 9.87   -> PV_SIGMA_FRAC ≈ 0.070
#   PP (PP marker):  base ≈ 0.221, k bound    -> PP profile is nearly flat;
#                                             PP marker is weakly zonated,
#                                             keep linear mode soft.
#   hex_side p25/p75 ≈ 196 / 294 level-2 px (real median ≈ 225)
#   noise (nuclear + membrane channels)/2:  mean 53–63, std 31–37
#
_CALIB_JSON = Path(__file__).resolve().parent / "synth_calibration_from_gt.json"
_CALIB: dict | None = None

# Module-level geometry cache: avoids recomputing the same (seed, hex_side)
# geometry when generate_all_instances is called multiple times (e.g. once for
# the convexity probe and once for the full stain generation).
_GEOM_CACHE: dict = {}   # key: (seed, hex_side) -> geom dict


def load_gt_calibration(path: Path | str | None = None) -> dict | None:
    """Load the JSON written by calibrate_synth_from_gt.py (or None if
    missing). Cached after first call."""
    global _CALIB
    if _CALIB is not None:
        return _CALIB
    p = Path(path) if path is not None else _CALIB_JSON
    if not p.exists():
        return None
    try:
        with open(p, "r") as fh:
            _CALIB = json.load(fh)
    except Exception as e:
        print(f"[synthetic_lobules] failed to load calibration {p}: {e}")
        return None
    return _CALIB


_calib = load_gt_calibration()
if _calib is not None:
    _rec = _calib.get("recommended_synthetic_params", {})
    # PV channel decay / baseline
    try:
        PP_BASELINE_CAL = float(_rec.get("PP_BASELINE"))
        PV_SIGMA_FRAC = float(_rec.get("PV_SIGMA_FRAC", PV_SIGMA_FRAC))
    except (TypeError, ValueError):
        PP_BASELINE_CAL = None
    # Autofluorescence is already baked into the fitted base, so drop it
    # to avoid adding twice.
    AUTOFLUO_FRAC = 0.0
    # Hex side range for generate_all_instances defaults
    _CAL_HEX_SIDES = tuple(int(x) for x in _rec.get("hex_sides", (220, 290)))
    if len(_CAL_HEX_SIDES) < 2:
        _CAL_HEX_SIDES = (220, 290)
    # Noise channel target stats
    _nmr = _rec.get("noise_target_mean_range", [55.0, 92.0])
    _nsr = _rec.get("noise_target_std_range", [28.0, 42.0])
    _CAL_NOISE_MEAN_RANGE = (float(_nmr[0]), float(_nmr[1]))
    _CAL_NOISE_STD_RANGE = (float(_nsr[0]), float(_nsr[1]))
    # Calibrated marker p95 ranges (real slides are dimmer than the old defaults)
    _c1r = _rec.get("cyp1_p95_range", [130.0, 160.0])
    _c3r = _rec.get("cyp3_p95_range", [ 70.0, 170.0])
    _CAL_CYP1_P95_RANGE = (float(_c1r[0]), float(_c1r[1]))
    _CAL_CYP3_P95_RANGE = (float(_c3r[0]), float(_c3r[1]))
    # Per-lobule expression multiplier (PV marker / PP marker mean / slide median)
    _epv = _rec.get("lobule_expr_pv", [0.55, 1.35])
    _epp = _rec.get("lobule_expr_pp", [0.50, 1.50])
    _CAL_LOBULE_EXPR_PV = (float(_epv[0]), float(_epv[1]))
    _CAL_LOBULE_EXPR_PP = (float(_epp[0]), float(_epp[1]))
    # Per-lobule PV shape diversity (base, k) - sampled per lobule
    _pbr = _rec.get("pv_base_range", [0.11, 0.22])
    _pkr = _rec.get("pv_k_range",    [6.5, 16.5])
    _CAL_PV_BASE_RANGE = (float(_pbr[0]), float(_pbr[1]))
    _CAL_PV_K_RANGE    = (float(_pkr[0]), float(_pkr[1]))
    # Pooled Hill parameters (shape model that won the comparison, if any)
    _CAL_BEST_SHAPE = str(_rec.get("best_shape_model", "exp"))
    _hill = _calib.get("fit", {}).get("pv", {}).get("models", {}).get("hill", {}).get("params", None)
    if _hill is not None and len(_hill) == 3:
        _CAL_HILL_PV = (float(_hill[0]), float(_hill[1]), float(_hill[2]))
    else:
        _CAL_HILL_PV = (0.16, 3.0, 0.6)
    # Replace the global-brightness range with something tied to the real
    # per-slide mean spread. Real slides vary absolute brightness ~3×, so
    # the old 0.35–1.0 multiplier stays sensible - but make sure we stay
    # inside the measured p95 envelope via the CYP p95 match downstream.
    GLOBAL_BRIGHTNESS_MIN = 0.6
    GLOBAL_BRIGHTNESS_MAX = 1.0
    # LOBULE_EXPR override: use the calibrated diversity
    LOBULE_EXPR_MIN = float(_CAL_LOBULE_EXPR_PV[0])
    LOBULE_EXPR_MAX = float(_CAL_LOBULE_EXPR_PV[1])
else:
    PP_BASELINE_CAL = None
    _CAL_HEX_SIDES = (300, 500)
    _CAL_NOISE_MEAN_RANGE = (55.0, 92.0)
    _CAL_NOISE_STD_RANGE = (28.0, 42.0)
    _CAL_CYP1_P95_RANGE = (180.0, 220.0)  # PV marker p95 range (uncalibrated fallback)
    _CAL_CYP3_P95_RANGE = (140.0, 220.0)  # PP marker p95 range (uncalibrated fallback)
    _CAL_LOBULE_EXPR_PV = (LOBULE_EXPR_MIN, LOBULE_EXPR_MAX)
    _CAL_LOBULE_EXPR_PP = (LOBULE_EXPR_MIN, LOBULE_EXPR_MAX)
    _CAL_PV_BASE_RANGE = (0.25, 0.35)
    _CAL_PV_K_RANGE = (2.0, 4.0)
    _CAL_BEST_SHAPE = "exp"
    _CAL_HILL_PV = (0.30, 3.0, 0.6)


# ---------------------------------------------------------------------------
#  Measured shape overrides (from output/compare_synth_vs_real_violin.py)
# ---------------------------------------------------------------------------
# The calibration JSON pinned the exp/hill amplitude at 1.0 and so inflated k.
# On the pooled *p99-normalized* real PV marker curve, with free amplitude, the
# best exponential fit is:
#     f(p) = base + (A - base) * exp(-k*(1-p))
#     base = 0.100,  A = 0.595,  k = 3.38   (R^2 = 0.994)
# The amplitude cap A < 1 is the quantitative signature of CV-label /
# PV-hotspot misalignment: the brightest pixel in a lobule is *not* exactly
# at the geometric CV. We reproduce this with (1) a per-lobule CV offset that
# shifts the effective peak inward from portality=1, and (2) a smooth spatial
# noise field on top of the portality map, so pixels near the CV end up with
# a spread of effective portalities rather than all sitting at exactly 1.
_CAL_PV_BASE_RANGE = (0.04, 0.10)          # measured per-lobule floor range
_CAL_PV_K_RANGE    = (2.5, 4.5)            # measured per-lobule decay range
PP_BASELINE_CAL    = 0.10                  # pooled base (exp fit)
# Measured amplitude cap on the *mean* stain curve at CV. Real pooled data
# with free amplitude gives A=0.60: the mean at CV-bin only reaches 60% of
# lobule p99. The extra 40% headroom is filled in by the log-normal
# cellular multiplier (bright cell clusters), so some individual pixels
# still reach ~1.0 (that is the per-lobule p99).
_CAL_PV_AMP_RANGE  = (0.50, 0.70)          # per-lobule amplitude cap A
# CV-jitter parameters (portality-space, applied inside generate_stain):
_CV_OFFSET_RANGE   = (0.00, 0.05)          # tiny peak softening only
_CV_SMOOTH_AMP     = 0.04                  # +/- amp of spatial jitter
_CV_SMOOTH_SIGMA   = 20.0                  # spatial corr length
# Intra-bin variance (log-normal cellular multiplier):
# Real PV marker has bright cell clusters reaching near-saturation alongside
# dim stromal gaps within the *same* portality bin. The Gaussian
# cell_texture (std ~0.20) is too tame to reproduce this. Use a log-normal
# field with sigma ~0.45 so ~5% of pixels reach 2-3x and ~5% drop to ~0.3x.
_CELL_LOGNORMAL_SIGMA = 1.00
_CELL_LOGNORMAL_CORR  = 2.0                # correlation length (px)


# ---------------------------------------------------------------------------
#  Tissue outline generation - fully procedural, no stored templates.
#
#  Parameters derived by fitting 60 real liver tissue sections
#  (tissue_outline_analysis.py). Edit the constants below to tune the
#  synthetic tissue shape statistics. No JSON / binary data needed.
# ---------------------------------------------------------------------------

# Number of EFD harmonics generated from scratch before fractal extension.
# Higher -> smoother base shape before fine detail is added.
TISSUE_N_HARM = 60

# Power-law exponent for harmonic amplitudes k ≥ 6: A(k) ∝ k^slope.
# More negative -> smoother edges; less negative -> rougher.
# Measured mean from 60 real sections was -1.47 (k=5..50 fit). We use -1.1
# so that higher-harmonic amplitudes remain large enough to be visible at
# practical canvas sizes (400–4000 px), giving genuine resolution-dependent
# boundary texture.
TISSUE_SLOPE_MEAN = -1.30
TISSUE_SLOPE_STD  =  0.18

# Normalised positive-harmonic amplitude envelope for k = 1 .. 10.
# k=1 is always 1.0 (normalization). Values at k=1..5 are medians from 60
# real sections; k=6..10 are boosted 1.5× relative to the measured medians
# so the power-law tail is large enough to produce visible pixel-level texture.
TISSUE_AMP_ENVELOPE = (
    1.000,   # k=1  - normalization reference
    0.502,   # k=2  - coarse shape
    0.592,   # k=3  - coarse shape (higher than k=2 in real liver sections)
    0.181,   # k=4  - intermediate
    0.130,   # k=5  - transition to power-law tail
    0.153,   # k=6  - ×1.5 vs measured median
    0.111,   # k=7
    0.104,   # k=8
    0.068,   # k=9
    0.071,   # k=10
)

# Per-harmonic amplitude standard deviation (Gaussian noise added per shape).
TISSUE_AMP_STD = (
    0.000,   # k=1  - fixed
    0.150,   # k=2
    0.180,   # k=3
    0.080,   # k=4
    0.060,   # k=5
    0.055,   # k=6
    0.045,   # k=7
    0.040,   # k=8
    0.030,   # k=9
    0.032,   # k=10
)

# Negative harmonic 1 amplitude ratio |c_{-1}| / |c_1|.
# Real liver sections are non-elliptical (crescent/S-shaped), so the negative
# fundamental is much larger than the positive. Median = 7.4 across 60
# sections; range 2.5–87. Drawn from LogNormal(log(median), sigma) per shape.
TISSUE_NEG1_RATIO_MEDIAN = 7.4
TISSUE_NEG1_RATIO_SIGMA  = 0.8    # log-normal sigma covering the 2.5–87 range

# Target tissue fraction range (fraction of canvas covered by tissue mask).
TISSUE_FRAC_MIN = 0.30
TISSUE_FRAC_MAX = 0.55


def _fd_to_contour(fd, n_harm, N, dc, scale, angle):
    """Reconstruct a closed contour from normalised Fourier descriptors.

    Mirrors the implementation in tissue_outline_analysis.py so the two
    scripts share the same convention without importing from each other.

    N must be > 2 * n_harm to avoid positive/negative harmonic overlap.
    """
    fd = np.asarray(fd, dtype=np.float64)
    pos = fd[:n_harm]             + 1j * fd[n_harm:2 * n_harm]
    neg = (fd[2 * n_harm:3 * n_harm] + 1j * fd[3 * n_harm:4 * n_harm])[::-1]
    Z_rot = np.zeros(N, dtype=complex)
    Z_rot[1:n_harm + 1]    = pos
    Z_rot[N - n_harm:N]    = neg
    freqs = np.fft.fftfreq(N) * N
    Z     = Z_rot * np.exp(1j * angle * freqs) * (N * scale)
    Z[0]  = N * (dc[0] + 1j * dc[1])
    z     = np.fft.ifft(Z)
    return np.stack([z.real, z.imag], axis=1)


# Feature size (pixels) used to decide how many harmonics are needed at a
# given canvas resolution. Smaller -> more harmonics -> finer boundary detail.
_FRACTAL_FEATURE_PX = 8
# Hard cap on the number of harmonics after fractal extension.
_FRACTAL_N_HARM_MAX = 800


def _extend_fd_fractal(fd, n_harm_stored, n_harm_target, slope, rng,
                       fd_amplitudes=None):
    """Extend FD to n_harm_target harmonics using fractal (power-law) synthesis.

    New harmonics are synthesised with amplitudes following the measured
    power-law spectrum  A(k) ∝ k^slope  (slope < 0, typically -1.4 … -1.8).
    Phases are drawn independently for each of the four components so the
    extension is stochastic but statistically consistent with real tissue.

    If ``fd_amplitudes`` is supplied (the per-harmonic amplitude profile stored
    in the JSON model), the local slope is re-fitted from the tail of the
    stored spectrum for more accurate extrapolation.

    The returned fd has length 4 * n_harm_target.
    N_reconst for _fd_to_contour must satisfy N_reconst > 2 * n_harm_target.
    """
    if n_harm_target <= n_harm_stored:
        return np.asarray(fd, dtype=np.float64), n_harm_stored

    fd_s = np.asarray(fd, dtype=np.float64)
    n0   = n_harm_stored
    n1   = n_harm_target

    # -- Amplitude reference ---------------------------------------------------
    if fd_amplitudes is not None and len(fd_amplitudes) >= n0:
        amps_stored = np.asarray(fd_amplitudes[:n0], dtype=np.float64)
    else:
        amps_stored = np.sqrt(fd_s[:n0] ** 2 + fd_s[n0:2 * n0] ** 2)

    # Fit local slope from the tail (last 25% of harmonics) for better
    # extrapolation when the spectrum deviates from a global power law.
    tail   = max(4, n0 // 4)
    k_tail = np.arange(n0 - tail + 1, n0 + 1, dtype=float)
    a_tail = np.clip(amps_stored[-tail:], 1e-12, None)
    try:
        local_slope = float(np.polyfit(np.log(k_tail), np.log(a_tail), 1)[0])
        local_slope = float(np.clip(local_slope, -3.5, -0.5))
    except Exception:
        local_slope = slope

    A_ref = float(amps_stored[-1]) if amps_stored[-1] > 1e-12 else 1e-4
    k_ref = float(n0)

    # -- Allocate extended layout [pos_re | pos_im | neg_re | neg_im] × n1 ----
    fd_ext = np.zeros(4 * n1, dtype=np.float64)
    fd_ext[:n0]          = fd_s[:n0]
    fd_ext[n1:n1 + n0]   = fd_s[n0:2 * n0]
    fd_ext[2*n1:2*n1+n0] = fd_s[2*n0:3*n0]
    fd_ext[3*n1:3*n1+n0] = fd_s[3*n0:4*n0]

    # -- Synthesise new harmonics n0+1 .. n1 ----------------------------------
    ks   = np.arange(n0 + 1, n1 + 1, dtype=float)
    amps = A_ref * (ks / k_ref) ** local_slope
    phi  = rng.uniform(0.0, 2.0 * np.pi, (4, len(ks)))

    idx = np.arange(n0, n1)
    fd_ext[idx]        = amps * np.cos(phi[0])
    fd_ext[n1 + idx]   = amps * np.sin(phi[1])
    fd_ext[2*n1 + idx] = amps * np.cos(phi[2])
    fd_ext[3*n1 + idx] = amps * np.sin(phi[3])

    return fd_ext, n1


# ---------------------------------------------------------------------------
#  Data class
# ---------------------------------------------------------------------------
@dataclass
class SyntheticInstance:
    """One synthetic tissue with all ground-truth and stain data."""
    seed: int
    hex_side: int
    stain_type: str       # label "pv_type+pp_type", e.g. "hill+linear"
    noise_level: str      # "low" or "high"
    mode: str             # "dual" or "single"

    tissue: np.ndarray = field(repr=False)
    gt_labels: np.ndarray = field(repr=False)        # (H,W) int32, 0=bg, >0=lobule id
    gt_portality: np.ndarray = field(repr=False)      # (H,W) float32, 0=boundary, 1=CV, NaN=bg
    central_mask: np.ndarray = field(repr=False)
    portal_mask: np.ndarray = field(repr=False)
    vessel_holes: np.ndarray = field(repr=False)
    gt_centers: np.ndarray = field(repr=False)        # (N,2) float64 (x,y)
    image_stack: np.ndarray = field(repr=False)        # (H,W,C) uint8 - the input to the pipeline
    pv_stain_raw: np.ndarray = field(repr=False)       # (H,W) float32 before uint8 conversion
    pp_stain_raw: np.ndarray = field(repr=False, default=None)
    shading_field: np.ndarray = field(repr=False, default=None)
    # (H,W) float32 in [0.5, 1.0] - the multiplicative flatfield-degradation
    # field applied to pv_stain_raw and pp_stain_raw.  Stored for preview.
    fused_pairs: list = field(repr=False, default_factory=list)
    # list of (label_a, label_b, v_e) for fused pairs
    fused_boundary_mask: np.ndarray = field(repr=False, default=None)
    # (H,W) bool - pixels on either side of every fused Voronoi edge;
    # None when there are no fusions.  Used by visualize_instances to
    # overlay the suppressed PP zone and by callers that need to know
    # where a shared boundary was artificially removed from GT.

    @property
    def fg(self):
        return self.tissue & ~self.vessel_holes

    @property
    def tag(self):
        return f"s{self.seed}_h{self.hex_side}_{self.stain_type}_{self.noise_level}_{self.mode}"


# ---------------------------------------------------------------------------
#  Tissue blob
# ---------------------------------------------------------------------------
def random_irregular_tissue(size, rng, n_pts=40, base_frac=0.38, noise_frac=0.12):
    """Single-blob tissue (legacy fallback, retained for callers)."""
    cx = cy = size / 2
    ang = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r0 = size * base_frac
    rad = r0 + (rng.random(n_pts) - 0.5) * (size * noise_frac)
    xs = cx + rad * np.cos(ang)
    ys = cy + rad * np.sin(ang)
    rr, cc = ski_polygon(ys, xs, (size, size))
    m = np.zeros((size, size), dtype=bool)
    m[rr, cc] = True
    return m


def fragmented_tissue(size, rng, target_frac=0.22, n_islands=None):
    """Multi-island irregular tissue mask mimicking real multiplexed fluorescence slices.

    Real sections are fragmented into many irregular clumps, not a single
    compact blob - tissue fraction across the 7 real images is 0.17–0.29,
    split across 3–8 disconnected islands of varying size.

    Parameters
    ----------
    size : int
        Canvas side length in pixels.
    rng : np.random.Generator
    target_frac : float
        Target fraction of the canvas covered by tissue (0.17–0.29 in real).
    n_islands : int or None
        Number of islands to place; if None, sampled from U[3, 8].
    """
    if n_islands is None:
        n_islands = int(rng.integers(1, 4))  # usually 1–3 islands

    m = np.zeros((size, size), dtype=bool)

    total_area = target_frac * (size ** 2)
    # Dominant island holds most of the area; satellites are small
    if n_islands > 1:
        weights = np.concatenate([
            [rng.uniform(0.65, 0.85)],
            rng.dirichlet(np.ones(n_islands - 1)) * rng.uniform(0.15, 0.35)
        ])
        weights = weights / weights.sum()
    else:
        weights = np.array([1.0])

    # Place main island near center, satellites scattered
    centers = [(size * rng.uniform(0.35, 0.65), size * rng.uniform(0.35, 0.65))]
    for _ in range(n_islands - 1):
        centers.append((size * rng.uniform(0.1, 0.9),
                         size * rng.uniform(0.1, 0.9)))

    def _junction_smooth_poly(xs, ys, pts_per_edge):
        """Apply per-vertex local smoothing to a closed polygon.

        Upsamples by pts_per_edge first so each edge has enough resolution,
        then applies the same sqrt-biased per-junction smoothing used for
        lobule boundaries (random extent 0–60%, random sigma 0–that extent).
        """
        n_coarse = len(xs)
        upsample = pts_per_edge
        N = n_coarse * upsample
        idx_f = np.linspace(0, n_coarse, N, endpoint=False)
        xf = np.interp(idx_f % n_coarse, np.arange(n_coarse), xs)
        yf = np.interp(idx_f % n_coarse, np.arange(n_coarse), ys)
        poly = np.stack([xf, yf], axis=1).copy()

        for vi in range(n_coarse):
            j = vi * upsample
            frac_L = 0.60 * float(rng.uniform(0.0, 1.0) ** 0.5)
            frac_R = 0.60 * float(rng.uniform(0.0, 1.0) ** 0.5)
            n_L = int(frac_L * upsample)
            n_R = int(frac_R * upsample)
            max_n = max(n_L, n_R)
            if max_n < 1:
                continue
            sigma = float(max_n) * float(rng.uniform(0.0, 1.0) ** 0.5)
            if sigma < 0.3:
                continue
            n_win = n_L + n_R + 1
            win_idx = np.arange(j - n_L, j + n_R + 1) % N
            window = poly[win_idx].copy()
            tiled_w = np.tile(window, (3, 1))
            sm_w = gaussian_filter1d(tiled_w.astype(np.float64),
                                     sigma=sigma, axis=0)[n_win:2 * n_win]
            dist = np.abs(np.arange(n_win) - n_L).astype(float)
            alpha = np.clip(np.cos(0.5 * np.pi * dist / float(max_n)),
                            0.0, 1.0)[:, None]
            poly[win_idx] = (1.0 - alpha) * poly[win_idx] + alpha * sm_w

        return poly[:, 0], poly[:, 1]

    for (cx, cy), w in zip(centers, weights):
        island_area = total_area * w
        n_pts = int(rng.integers(26, 48))
        ang = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        ang += rng.uniform(0, 2 * np.pi)
        r0 = np.sqrt(island_area / np.pi)
        ax = r0 * rng.uniform(0.85, 1.20)
        ay = r0 * rng.uniform(0.85, 1.20)
        rnoise = rng.uniform(0.72, 1.28, n_pts)
        xs = cx + ax * rnoise * np.cos(ang)
        ys = cy + ay * rnoise * np.sin(ang)
        xs, ys = _junction_smooth_poly(xs, ys, pts_per_edge=6)
        xs = np.clip(xs, 0, size - 1)
        ys = np.clip(ys, 0, size - 1)
        rr, cc = ski_polygon(ys, xs, (size, size))
        m[rr, cc] = True

    # Optional: one random interior tear, simulating a real sectioning artifact.
    if rng.random() < 0.6:
        ys_t, xs_t = np.where(m)
        if len(ys_t) > 500:
            t_idx = int(rng.integers(0, len(ys_t)))
            hy, hx = ys_t[t_idx], xs_t[t_idx]
            hr = rng.uniform(15, 55)
            n_pts_h = int(rng.integers(12, 22))
            hang = np.linspace(0, 2 * np.pi, n_pts_h, endpoint=False)
            hr_v = hr * rng.uniform(0.55, 1.45, n_pts_h)
            hxs = hx + hr_v * np.cos(hang)
            hys = hy + hr_v * np.sin(hang)
            hxs, hys = _junction_smooth_poly(hxs, hys, pts_per_edge=6)
            hxs = np.clip(hxs, 0, size - 1)
            hys = np.clip(hys, 0, size - 1)
            rr, cc = ski_polygon(hys, hxs, (size, size))
            m[rr, cc] = False

    # Light open/close to remove stray pixels.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(bool)

    return m


def tissue_from_outline(size: int, rng, target_frac: float | None = None) -> np.ndarray:
    """Generate an organic tissue outline mask by procedural Fourier synthesis.

    Builds Elliptic Fourier Descriptors (EFD) from scratch using statistics
    measured from 60 real liver tissue sections. No stored templates or binary
    data - everything is derived from the module-level TISSUE_* constants.

    The fundamental shape is an ellipse (aspect ratio ~1.3, from the large
    negative harmonic 1) with fractal boundary texture added via power-law
    amplitude extension. Random phases give unique organic shapes per call.

    Parameters
    ----------
    size : int
        Output canvas side length (== IMG).
    rng : np.random.Generator
    target_frac : float or None
        Target tissue coverage as fraction of canvas.
        Drawn from U[TISSUE_FRAC_MIN, TISSUE_FRAC_MAX] when None.
    """
    if target_frac is None:
        target_frac = float(rng.uniform(TISSUE_FRAC_MIN, TISSUE_FRAC_MAX))

    n_harm = TISSUE_N_HARM
    n_env  = len(TISSUE_AMP_ENVELOPE)

    # -- Sample per-shape parameters ------------------------------------------
    slope = float(np.clip(
        rng.normal(TISSUE_SLOPE_MEAN, TISSUE_SLOPE_STD), -2.5, -0.4))

    # -- Positive harmonic amplitudes (amps_pos[i] = |c_{i+1}|) ---------------
    amps_pos = np.zeros(n_harm)
    amps_pos[0] = 1.0   # k=1 normalised

    for i in range(1, min(n_env, n_harm)):
        amps_pos[i] = max(0.005,
                          float(rng.normal(TISSUE_AMP_ENVELOPE[i],
                                           TISSUE_AMP_STD[i])))

    # Power law for k > n_env: anchor at last envelope point
    if n_harm > n_env:
        k_anch = float(n_env)                         # 1-indexed harmonic
        A_anch = max(amps_pos[n_env - 1], 1e-6)
        A0_pl  = A_anch / (k_anch ** slope)
        for i in range(n_env, n_harm):
            amps_pos[i] = max(1e-6, A0_pl * ((i + 1.0) ** slope))

    # -- Negative harmonic amplitudes (neg_amps_k[i] = |c_{-(i+1)}|) ----------
    # In _fd_to_contour, fd[2*n:3*n][i] = Re(c_{-(i+1)}) (derived from the
    # IFFT index mapping: DFT slot N-(n-j) ↔ frequency -(n-j), and after the
    # internal reversal, slot 0 of neg_re corresponds to c_{-1}).
    neg_amps_k = np.zeros(n_harm)
    # c_{-1}: dominant negative harmonic - real liver shapes are non-elliptical
    neg1_ratio = float(np.clip(
        rng.lognormal(np.log(TISSUE_NEG1_RATIO_MEDIAN),
                      TISSUE_NEG1_RATIO_SIGMA),
        1.0, 200.0))
    neg_amps_k[0] = neg1_ratio   # k=1 large

    # c_{-k} for k≥2: same power law as pos, independent random scale factor
    for i in range(1, n_harm):
        neg_amps_k[i] = amps_pos[i] * float(rng.lognormal(0.0, 0.3))

    # -- Random phases ---------------------------------------------------------
    phi_pos = rng.uniform(0.0, 2.0 * np.pi, n_harm)
    phi_neg = rng.uniform(0.0, 2.0 * np.pi, n_harm)

    pos_c     = amps_pos  * np.exp(1j * phi_pos)
    neg_c_by_k = neg_amps_k * np.exp(1j * phi_neg)  # neg_c_by_k[i] = c_{-(i+1)}

    # Force c_1 real positive (fixes orientation normalisation)
    pos_c[0] = 1.0 + 0j

    # -- Pack FD: layout [pos_re | pos_im | neg_re | neg_im] each length n_harm
    # fd[2*n+i] = Re(c_{-(i+1)}), so neg_c_by_k is stored as-is (no reversal).
    fd = np.concatenate([
        pos_c.real, pos_c.imag,
        neg_c_by_k.real, neg_c_by_k.imag,
    ])

    # -- Draw aug_angle and flip BEFORE fractal extension so they are
    # resolution-independent (extension consumes a variable number of rng
    # calls depending on size, which would otherwise shift the flip coin).
    aug_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    do_flip   = rng.random() < 0.5

    # -- Scale-adaptive fractal extension (same logic as old template path) ----
    est_perim   = np.pi * np.sqrt(4.0 * target_frac) * size * 1.5
    n_harm_need = min(int(est_perim / _FRACTAL_FEATURE_PX), _FRACTAL_N_HARM_MAX)
    n_harm_need = max(n_harm_need, n_harm)
    if n_harm_need > n_harm:
        fd, n_harm_use = _extend_fd_fractal(fd, n_harm, n_harm_need, slope, rng)
    else:
        n_harm_use = n_harm

    n_reconst = max(4096, int(est_perim * 0.5))

    # -- Reconstruct contour ---------------------------------------------------
    pts = _fd_to_contour(fd, n_harm_use, n_reconst,
                         dc=np.zeros(2), scale=0.5, angle=aug_angle)

    template_area = abs(cv2.contourArea(pts.astype(np.float32)))
    if template_area < 1e-3:
        return fragmented_tissue(size, rng)

    # Optional horizontal flip for extra variety
    if do_flip:
        pts[:, 0] = -pts[:, 0]

    # -- Scale to target coverage, capped so bounding box fits canvas ----------
    sf       = np.sqrt(target_frac * size * size / template_area)
    bw       = (pts[:, 0].max() - pts[:, 0].min()) * sf
    bh       = (pts[:, 1].max() - pts[:, 1].min()) * sf
    margin   = 0.02 * size
    max_span = size - 2 * margin
    if max(bw, bh) > max_span:
        sf *= max_span / max(bw, bh)

    # -- Centre on canvas -----------------------------------------------------
    pts_f   = pts * sf + np.array([size / 2.0, size / 2.0])
    pts_int = np.clip(pts_f, 0, size - 1).astype(np.int32).reshape(-1, 1, 2)

    # -- Rasterize; take largest exterior contour ------------------------------
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int], 1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        mask2 = np.zeros_like(mask)
        cv2.fillPoly(mask2, [max(cnts, key=cv2.contourArea)], 1)
        mask = mask2

    k_sz = max(3, size // 300) | 1   # odd: ~5 at 1500px, ~13 at 4000px
    k5   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    result = mask.astype(bool)

    if result.mean() < 0.05:
        return fragmented_tissue(size, rng)

    # Validity: reject shapes that are too elongated or too hollow
    # (self-intersecting contours that fragmented into a narrow filament).
    ys_r, xs_r = np.where(result)
    bh = int(ys_r.max() - ys_r.min()) + 1
    bw = int(xs_r.max() - xs_r.min()) + 1
    fill   = result.sum() / (bh * bw)          # solidity proxy
    aspect = max(bh, bw) / max(1, min(bh, bw)) # long-axis / short-axis
    if fill < 0.30 or aspect > 4.5:
        return fragmented_tissue(size, rng)

    return result


def generate_tissue_fold(shape, tissue, rng):
    """Generate a bright linear tissue fold artifact.

    Returns a float32 mask with fold intensity [0, 1].
    Real tissue folds appear as bright streaks where the section
    is doubled-over, causing 2x fluorescence.
    """
    h, w = shape
    fold = np.zeros((h, w), dtype=np.float32)

    # Random line across the tissue
    # Pick two points on tissue boundary
    ys, xs = np.where(tissue)
    if len(ys) < 100:
        return fold

    # Start and end on tissue edge
    idx1, idx2 = rng.choice(len(ys), 2, replace=False)
    y1, x1 = float(ys[idx1]), float(xs[idx1])
    y2, x2 = float(ys[idx2]), float(xs[idx2])

    # Draw thick line with Gaussian cross-section
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length < 50:
        return fold

    # Width varies (10-40 px)
    fold_width = rng.uniform(10, 40)

    # Create distance field to the line
    # Line parameterized as p1 + t*(p2-p1), t in [0,1]
    dx, dy = x2 - x1, y2 - y1
    yy, xx = np.ogrid[0:h, 0:w]
    # Project each pixel onto line
    t = ((xx - x1) * dx + (yy - y1) * dy) / (length**2 + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    # Distance from pixel to nearest point on line
    px = x1 + t * dx
    py = y1 + t * dy
    dist = np.sqrt((xx - px)**2 + (yy - py)**2)

    # Gaussian profile
    fold = np.exp(-0.5 * (dist / fold_width)**2).astype(np.float32)
    fold *= FOLD_INTENSITY
    fold[~tissue] = 0.0

    return fold


def _smooth_lobule_map(m: np.ndarray, tissue: np.ndarray, sigma: float,
                        fill: float) -> np.ndarray:
    """Tissue-aware Gaussian blur for per-lobule scalar maps.

    Per-lobule maps paint a constant inside each lobule which creates a
    hard step at lobule boundaries. This smooths the map with a Gaussian
    kernel, normalising by the tissue weight so values don't bleed into
    the background, then restores the fill value outside tissue.
    """
    if sigma <= 0:
        return m
    w = tissue.astype(np.float32)
    num = gaussian_filter((m.astype(np.float32) * w), sigma=sigma)
    den = gaussian_filter(w, sigma=sigma) + 1e-8
    out = num / den
    out[~tissue] = fill
    return out.astype(np.float32)


def generate_lobule_expression_map(labels, kept_ids, rng,
                                     lo: float | None = None,
                                     hi: float | None = None):
    """Per-lobule expression multiplier to model biological heterogeneity.

    Default range comes from the GT calibration
    `recommended_synthetic_params.lobule_expr_pv`.
    """
    if lo is None:
        lo = LOBULE_EXPR_MIN
    if hi is None:
        hi = LOBULE_EXPR_MAX
    h, w = labels.shape
    expr_map = np.ones((h, w), dtype=np.float32)
    for lid in kept_ids:
        mult = rng.uniform(float(lo), float(hi))
        expr_map[labels == lid] = mult
    return expr_map


def generate_lobule_base_map(labels, kept_ids, rng,
                               lo: float | None = None,
                               hi: float | None = None):
    """Per-lobule PV baseline (floor of the stain curve) map.

    The real GT shows per-lobule bases spanning ~0.11–0.22 (p25–p75) for
    the PV marker, which the single global `PP_BASELINE` cannot capture. This
    samples an independent base per lobule so the synthetic per-lobule
    base distribution matches the real one.
    """
    if lo is None:
        lo = _CAL_PV_BASE_RANGE[0]
    if hi is None:
        hi = _CAL_PV_BASE_RANGE[1]
    h, w = labels.shape
    base_map = np.full((h, w), 0.5 * (lo + hi), dtype=np.float32)
    for lid in kept_ids:
        base_map[labels == lid] = float(rng.uniform(lo, hi))
    return base_map


def generate_lobule_k_map(labels, kept_ids, rng,
                           lo: float | None = None,
                           hi: float | None = None):
    """Per-lobule PV sharpness (decay rate) map."""
    if lo is None:
        lo = _CAL_PV_K_RANGE[0]
    if hi is None:
        hi = _CAL_PV_K_RANGE[1]
    h, w = labels.shape
    k_map = np.full((h, w), 0.5 * (lo + hi), dtype=np.float32)
    for lid in kept_ids:
        k_map[labels == lid] = float(rng.uniform(lo, hi))
    return k_map


def generate_lobule_amp_map(labels, kept_ids, rng,
                              lo: float | None = None,
                              hi: float | None = None):
    """Per-lobule PV amplitude cap (A in `base + (A-base)*shape`).

    On real pooled p99-normalized PV marker data, fitting with free amplitude
    gives A≈0.60 - i.e. the *mean* stain intensity at CV reaches only ~60%
    of per-lobule p99. The remaining headroom is filled by bright cellular
    clusters (log-normal texture), not by the shape itself.
    """
    if lo is None:
        lo = _CAL_PV_AMP_RANGE[0]
    if hi is None:
        hi = _CAL_PV_AMP_RANGE[1]
    h, w = labels.shape
    amp_map = np.full((h, w), 0.5 * (lo + hi), dtype=np.float32)
    for lid in kept_ids:
        amp_map[labels == lid] = float(rng.uniform(lo, hi))
    return amp_map


def generate_cv_offset_map(labels, kept_ids, rng,
                            lo: float | None = None,
                            hi: float | None = None):
    """Per-lobule CV-portality offset map.

    Each lobule gets a scalar offset in [lo, hi]. Later, the portality map
    used by the PV stain shape is replaced by ``P_eff = clip(P - offset +
    spatial_noise, 0, 1)`` so the effective stain peak is not pinned exactly
    to the geometric central vein. This reproduces the "ceiling below 1.0"
    observed on real PV marker data (amplitude cap ~0.6 of per-lobule p99).
    """
    if lo is None:
        lo = _CV_OFFSET_RANGE[0]
    if hi is None:
        hi = _CV_OFFSET_RANGE[1]
    h, w = labels.shape
    off_map = np.zeros((h, w), dtype=np.float32)
    for lid in kept_ids:
        off_map[labels == lid] = float(rng.uniform(lo, hi))
    return off_map


# ---------------------------------------------------------------------------
#  Hex grid
# ---------------------------------------------------------------------------
def hex_grid_centers(size, side, rng, jitter_frac=0.55):
    dx = 1.5 * side
    dy = math.sqrt(3.0) * side
    out = []
    for i in range(int(size / dx) + 3):
        for j in range(int(size / dy) + 3):
            x = i * dx + rng.uniform(-jitter_frac * side, jitter_frac * side)
            y = (j * dy + ((i & 1) * 0.5 * dy)
                 + rng.uniform(-jitter_frac * side, jitter_frac * side))
            if 0 <= x < size and 0 <= y < size:
                out.append((x, y))
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------------------
#  Voronoi -> perturbed lobule polygons
# ---------------------------------------------------------------------------
def _perturb_edge(p0, p1, rng, hex_side):
    """Create a curved path between p0 and p1.

    Each edge is independently drawn as either:
      - Single arch  (50 %): one Gaussian bump, random side and position
      - Double S     (50 %): two bumps with opposite amplitudes placed at
                             ~1/3 and ~2/3 of the edge, creating an S-curve

    Displacement profile:
        d(t) = sum_i  A_i * exp(-0.5 * ((t - t_i) / sigma_t)^2)

    Parameters drawn per edge:
        |A|     ~ HalfNormal(sigma=0.45 * max_ortho), clipped at max_ortho
        sigma_t ~ Uniform[0.10, 0.22]   (width of each bump)
        t_i       placed at thirds for S, random centre for single arch
    """
    length = np.linalg.norm(p1 - p0)
    if length < 1.0:
        return np.array([p0, p1])

    tangent = (p1 - p0) / length
    normal  = np.array([-tangent[1], tangent[0]])

    max_ortho = PERTURB_ORTHO_FRAC * length
    sigma_A   = 0.50 * max_ortho   # typical displacement ~40% of max
    sigma_t   = float(rng.uniform(0.15, 0.35))  # wider bumps -> gentler curves

    # Amplitude magnitude (always positive; sign assigned below)
    amp = float(np.clip(abs(rng.normal(0.0, sigma_A)), 0.0, max_ortho))

    if rng.random() < 0.5:
        # --- Single arch ---
        A  = amp * (1 if rng.random() < 0.5 else -1)
        t0 = float(rng.uniform(0.25, 0.75))
        bumps = [(A, t0)]
    else:
        # --- Double S: two bumps, opposite sign, at ~1/3 and ~2/3 ---
        sign = 1 if rng.random() < 0.5 else -1
        t1   = float(rng.uniform(0.20, 0.38))
        t2   = float(rng.uniform(0.62, 0.80))
        bumps = [(sign * amp, t1), (-sign * amp, t2)]

    # Build N_total points: endpoints + PERTURB_N_PTS intermediate
    N_total = PERTURB_N_PTS + 2
    t_vals  = np.linspace(0.0, 1.0, N_total)

    pts = np.empty((N_total, 2), dtype=np.float64)
    pts[0]  = p0
    pts[-1] = p1
    for k in range(1, N_total - 1):
        t    = t_vals[k]
        base = p0 + t * (p1 - p0)
        d    = sum(A_i * np.exp(-0.5 * ((t - t_i) / sigma_t) ** 2)
                   for A_i, t_i in bumps)
        pts[k] = base + d * normal

    return pts


def _segments_intersect(a0, a1, b0, b1):
    """Check if line segment a0-a1 intersects b0-b1 (proper crossing)."""
    d1 = (b1[0] - b0[0]) * (a0[1] - b0[1]) - (b1[1] - b0[1]) * (a0[0] - b0[0])
    d2 = (b1[0] - b0[0]) * (a1[1] - b0[1]) - (b1[1] - b0[1]) * (a1[0] - b0[0])
    d3 = (a1[0] - a0[0]) * (b0[1] - a0[1]) - (a1[1] - a0[1]) * (b0[0] - a0[0])
    d4 = (a1[0] - a0[0]) * (b1[1] - a0[1]) - (a1[1] - a0[1]) * (b1[0] - a0[0])
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return False


def _polygon_self_intersects(pts):
    """Check if a polygon (array of vertices) has any self-intersecting edges."""
    n = len(pts)
    for i in range(n):
        a0 = pts[i]
        a1 = pts[(i + 1) % n]
        # Check against non-adjacent edges
        for j in range(i + 2, n):
            if j == (i - 1) % n or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex
            b0 = pts[j]
            b1 = pts[(j + 1) % n]
            if _segments_intersect(a0, a1, b0, b1):
                return True
    return False


def build_perturbed_voronoi_labels(centers, tissue, hex_side, rng):
    """Build connected lobule labels from Voronoi + boundary perturbation.

    Each Voronoi ridge is perturbed exactly once using a deterministic per-ridge
    sub-rng (derived from a single value consumed from the main rng).  Both
    adjacent polygons reuse the same pre-computed perturbed points - one forward,
    the other reversed - so their shared boundary is identical and no mismatch
    strip exists between them.  This eliminates interlocking without affecting the
    main rng state for the junction-smoothing loop that follows.

    Gaps at polygon vertices (where ≥3 polygons meet) are filled using the
    original EDT-from-existing-region approach, which interpolates the nearby
    curved boundaries and preserves the organic shape appearance.

    If a polygon self-intersects after perturbation, it falls back to its
    straight (unperturbed) Voronoi cell.

    Returns (labels, kept_ids) where labels is (H,W) int32.
    """
    h, w = tissue.shape

    # Add mirror points to bound the Voronoi diagram
    mirror = []
    for cx, cy in centers:
        mirror.append((-cx, cy))
        mirror.append((2 * w - cx, cy))
        mirror.append((cx, -cy))
        mirror.append((cx, 2 * h - cy))
    all_pts = np.vstack([centers, np.array(mirror)])

    vor = Voronoi(all_pts)

    # Jitter Voronoi vertices (corners) globally - shared across all
    # adjacent polygons so lobules still tile without gaps
    jitter_max = VERTEX_JITTER_FRAC * hex_side
    jittered_vertices = vor.vertices.copy()
    jittered_vertices += rng.uniform(-jitter_max, jitter_max,
                                      size=jittered_vertices.shape)

    n_real = len(centers)

    # Pre-compute ONE perturbed edge per Voronoi ridge using a per-ridge
    # deterministic sub-rng.  Consuming exactly 1 value from the main rng
    # (for _ridge_seed) keeps the junction-smoothing loop below on a
    # consistent rng trajectory regardless of the number of ridges.
    ridge_pts: dict = {}
    ridge_by_verts: dict = {}
    _ridge_seed = int(rng.integers(2 ** 32))
    for k, (v_p, v_q) in enumerate(vor.ridge_vertices):
        if v_p < 0 or v_q < 0:
            continue
        ridge_by_verts[(min(v_p, v_q), max(v_p, v_q))] = k
        _sub = np.random.default_rng(
            _ridge_seed ^ (int(min(v_p, v_q)) * 1_000_003
                           + int(max(v_p, v_q)) * 999_983)
        )
        ridge_pts[k] = _perturb_edge(
            jittered_vertices[v_p], jittered_vertices[v_q], _sub, hex_side
        )

    # For each real region, build a perturbed polygon
    labels = np.zeros((h, w), dtype=np.int32)

    # Step 1: collect perturbed polygons for real centres
    polys = {}
    n_rejected = 0
    for idx in range(n_real):
        reg_idx = vor.point_region[idx]
        region = vor.regions[reg_idx]
        if -1 in region or len(region) < 3:
            continue
        verts = jittered_vertices[region]

        # Assemble polygon from the pre-computed shared ridge points.
        perturbed_pts = []
        for i in range(len(region)):
            v_i = region[i]
            v_j = region[(i + 1) % len(region)]
            key = (min(v_i, v_j), max(v_i, v_j))
            k = ridge_by_verts.get(key)
            if k is not None:
                pts = ridge_pts[k]               # v_p -> v_q, includes endpoints
                rp = vor.ridge_vertices[k][0]
                if v_i == rp:
                    edge_pts = pts[:-1]           # forward, drop last
                else:
                    edge_pts = pts[::-1][:-1]     # reversed
            else:
                edge_pts = jittered_vertices[[v_i]]   # infinite ridge fallback
            perturbed_pts.append(edge_pts)
        poly = np.vstack(perturbed_pts)

        # Check for self-intersection; fall back to straight cell if needed.
        if _polygon_self_intersects(poly):
            poly = verts.copy()
            n_rejected += 1
        else:
            # Per-junction local smoothing: at each vertex where two curved
            # edges meet, independently choose how much of each adjacent edge
            # gets rounded and with what sigma.
            #   frac_L, frac_R ~ U[0, 0.6]  -> smoothed fraction of each side
            #   sigma           ~ U[0, max(n_L, n_R)]  -> in point-index units
            # A cosine taper blends smoothed->original at the window edges so
            # there is no hard discontinuity between smoothed and raw sections.
            n_verts = len(verts)
            pts_per_edge = PERTURB_N_PTS + 1   # points contributed per edge
            N = len(poly)
            smoothed_poly = poly.copy()

            for vi in range(n_verts):
                j = (vi * pts_per_edge) % N     # junction index in poly

                # Square-root sampling biases toward the upper end of each
                # range: U^0.5 has PDF 2u so large fracs/sigmas are ~2x more
                # likely than small ones -> most junctions are visibly rounded.
                frac_L = 0.60 * float(rng.uniform(0.0, 1.0) ** 0.5)
                frac_R = 0.60 * float(rng.uniform(0.0, 1.0) ** 0.5)
                n_L = int(frac_L * pts_per_edge)
                n_R = int(frac_R * pts_per_edge)
                max_n = max(n_L, n_R)
                if max_n < 1:
                    continue
                sigma = float(max_n) * float(rng.uniform(0.0, 1.0) ** 0.5)
                if sigma < 0.3:
                    continue

                n_win = n_L + n_R + 1
                win_idx = np.arange(j - n_L, j + n_R + 1) % N
                window = smoothed_poly[win_idx].copy()

                # Gaussian smooth the local window
                tiled_w = np.tile(window, (3, 1))
                sm_w = gaussian_filter1d(tiled_w.astype(np.float64),
                                         sigma=sigma, axis=0)
                sm_w = sm_w[n_win: 2 * n_win]

                # Cosine taper: alpha=1 at junction centre, 0 at window edges
                pos = np.arange(n_win)
                dist = np.abs(pos - n_L).astype(float)
                alpha = np.cos(0.5 * np.pi * dist / float(max_n))
                alpha = np.clip(alpha, 0.0, 1.0)[:, None]

                smoothed_poly[win_idx] = (1.0 - alpha) * smoothed_poly[win_idx] + alpha * sm_w

            if not _polygon_self_intersects(smoothed_poly):
                poly = smoothed_poly

        polys[idx] = poly

    # Step 2: rasterize all polygons (may overlap at perturbed boundaries)
    # Use a coverage-count array to detect overlaps
    coverage = np.zeros((h, w), dtype=np.int32)  # count of polygons covering each pixel

    for idx, poly in polys.items():
        label_id = idx + 1
        # Clip polygon to image bounds
        poly_clipped = poly.copy()
        poly_clipped[:, 0] = np.clip(poly_clipped[:, 0], 0, w - 1)
        poly_clipped[:, 1] = np.clip(poly_clipped[:, 1], 0, h - 1)
        if len(poly_clipped) < 3:
            continue
        rr, cc = ski_polygon(poly_clipped[:, 1], poly_clipped[:, 0], (h, w))
        labels[rr, cc] = label_id
        coverage[rr, cc] += 1

    # Step 3: for overlap pixels (coverage > 1), assign to nearest centre
    overlap = coverage > 1
    if np.any(overlap):
        oy, ox = np.where(overlap)
        dists = np.zeros((len(oy), n_real), dtype=np.float64)
        for i in range(n_real):
            dx = ox - centers[i, 0]
            dy = oy - centers[i, 1]
            dists[:, i] = dx * dx + dy * dy
        nearest = np.argmin(dists, axis=1) + 1  # label_id = idx + 1
        labels[oy, ox] = nearest.astype(np.int32)

    # Step 4: fill gaps (coverage == 0 inside tissue) with nearest-region EDT.
    # With shared-ridge perturbation, gaps only arise at polygon vertices
    # (where 3+ polygons meet) and at the tissue boundary - never between
    # adjacent polygon interiors.  EDT-from-existing-region correctly
    # interpolates these tiny vertex gaps following the local curved shapes.
    gap = tissue & (labels == 0)
    if np.any(gap):
        min_d = np.full((h, w), np.inf, dtype=np.float64)
        for idx in range(n_real):
            label_id = idx + 1
            if not np.any(labels == label_id):
                continue
            d = edt(labels != label_id)
            closer = gap & (d < min_d)
            labels[closer] = label_id
            min_d[closer] = d[closer]

    # Zero out non-tissue
    labels[~tissue] = 0

    # Discard tiny / edge lobules (these would otherwise leave dark cutouts).
    # Two criteria:
    #   1. Area < 15 % of a full hexagonal cell - lobule too small.
    #   2. Bbox fill < 0.25 - lobule is a degenerate thin sliver (wedge
    #      formed when the tissue boundary clips a Voronoi cell at an acute
    #      angle).  A normal lobule fills ≥ 40 % of its bounding box.
    kept_ids = []
    for label_id in range(1, n_real + 1):
        _mask = labels == label_id
        area = int(_mask.sum())
        if area < 0.15 * (hex_side ** 2):
            labels[_mask] = 0
            continue
        _ys, _xs = np.where(_mask)
        _bbox_area = int((_ys.max() - _ys.min() + 1) * (_xs.max() - _xs.min() + 1))
        if _bbox_area > 0 and area / _bbox_area < 0.25:   # sliver check
            labels[_mask] = 0
            continue
        kept_ids.append(label_id)

    # Re-fill any holes left behind by the discard step (EDT from kept regions).
    gap2 = tissue & (labels == 0)
    if np.any(gap2) and kept_ids:
        min_d2 = np.full((h, w), np.inf, dtype=np.float64)
        for label_id in kept_ids:
            d = edt(labels != label_id)
            closer = gap2 & (d < min_d2)
            labels[closer] = label_id
            min_d2[closer] = d[closer]

    # Step 5: enforce simply-connected labels.
    # Despite shared ridge perturbation, straight-polygon fallbacks can leave
    # tiny disconnected slivers.  Absorb every secondary component of each
    # label into its dominant 8-connected neighbour - regardless of size -
    # so the final GT has no interlocking regions.
    _conn8 = _ndstruct(2, 2)
    for label_id in list(kept_ids):
        _cc_map, _n_cc = _ndlabel(labels == label_id, structure=_conn8)
        if _n_cc <= 1:
            continue
        _sizes = np.bincount(_cc_map.ravel())      # index 0 = not this label
        _main_cc = int(np.argmax(_sizes[1:]) + 1)  # largest component
        for _cc_id in range(1, _n_cc + 1):
            if _cc_id == _main_cc:
                continue
            _frag = _cc_map == _cc_id
            # Dilate 1 px to find touching neighbours
            _nbr = _nddilate(_frag, structure=_conn8) & (labels > 0) & ~_frag
            if _nbr.any():
                _vals, _cnts = np.unique(labels[_nbr], return_counts=True)
                labels[_frag] = _vals[np.argmax(_cnts)]
            else:
                labels[_frag] = 0

    return labels, kept_ids


# ---------------------------------------------------------------------------
#  Vessels
# ---------------------------------------------------------------------------
def paint_ellipse_mask(mask, cx, cy, rx, ry):
    H, W = mask.shape
    y0 = max(0, int(cy - ry) - 1)
    y1 = min(H, int(cy + ry) + 2)
    x0 = max(0, int(cx - rx) - 1)
    x1 = min(W, int(cx + rx) + 2)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= ((xx - cx) ** 2 / rx ** 2
                             + (yy - cy) ** 2 / ry ** 2 <= 1.0)


def paint_blob_mask(mask, cx, cy, rx, ry, rng, n_pts=None, rnoise_range=(0.55, 1.45)):
    """Paint an irregular blob (random-radius polygon) into ``mask``.

    Similar silhouette to ``paint_ellipse_mask`` but with per-vertex radius
    noise so the boundary is organic rather than elliptical.

    rnoise_range widened to (0.30, 1.70) so blobs have realistic circularity
    (0.2–0.6) matching real CV/PP vessel cross-sections that include oblique
    sections, partial collapse, and irregular lumens.  The optimizer then learns
    vessel_circularity_min values that transfer to real data.
    """
    H, W = mask.shape
    if n_pts is None:
        n_pts = int(rng.integers(12, 22))   # more vertices -> smoother yet still irregular
    ang = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ang += rng.uniform(0, 2 * np.pi)
    # Optionally rotate the whole ellipse to simulate oblique sections
    rot = rng.uniform(0, np.pi)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rnoise = rng.uniform(rnoise_range[0], rnoise_range[1], n_pts)
    # Local axes before rotation
    lx = rx * rnoise * np.cos(ang)
    ly = ry * rnoise * np.sin(ang)
    xs = cx + cos_r * lx - sin_r * ly
    ys = cy + sin_r * lx + cos_r * ly
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)
    rr, cc = ski_polygon(ys, xs, (H, W))
    mask[rr, cc] = True


def place_vessels(centers, kept_ids, labels, tissue, hex_side, rng):
    """Place CV ellipses at lobule centroids and portal triads at Voronoi vertices."""
    h, w = tissue.shape
    central_mask = np.zeros((h, w), dtype=bool)
    portal_mask = np.zeros((h, w), dtype=bool)
    gt_centers_out = []

    # Minimum distance the CV centre must be from the lobule boundary.
    # Ensures the CV sits well inside the lobule even for edge-clipped cells.
    # Set to max possible CV radius + a safety margin.
    _CV_MIN_DIST = float(max(CV_RX[1], CV_RY[1])) + 8.0

    # CV: one ellipse per kept lobule, placed at the deepest interior point
    # (maximum of the distance transform) rather than the centroid.  This
    # guarantees the CV is as far from all boundaries as possible.
    for label_id in kept_ids:
        _lob_mask = labels == label_id
        if not _lob_mask.any():
            continue
        # Distance of every lobule pixel from the lobule boundary.
        _dist = edt(_lob_mask)
        _max_dist = float(_dist.max())
        # Deepest interior point - use this as the CV centre.
        _flat = int(np.argmax(_dist))
        cy, cx = float(_flat // w), float(_flat % w)
        gt_centers_out.append((cx, cy))
        if rng.random() < CV_DROPOUT:
            continue  # this lobule has no central vein
        if _max_dist < _CV_MIN_DIST:
            continue  # lobule too thin to place a realistic CV
        rx = rng.integers(CV_RX[0], CV_RX[1] + 1)
        ry = rng.integers(CV_RY[0], CV_RY[1] + 1)
        paint_blob_mask(central_mask, cx, cy, rx, ry, rng)

    # PP triads: at boundary junctions (Voronoi vertices)
    # Find boundary pixels where 3+ different labels meet
    # Use dilation to find vertex-like junction regions
    vertex_points = _find_label_junctions(labels, kept_ids)

    for vx, vy in vertex_points:
        if not (0 <= int(vx) < w and 0 <= int(vy) < h and tissue[int(vy), int(vx)]):
            continue
        # Random dropout
        if rng.random() < PP_DROPOUT:
            continue
        # Place 1-3 vessels (triad)
        n_vessels = rng.integers(1, PP_TRIAD_MAX + 1)
        for _ in range(n_vessels):
            jx = vx + rng.uniform(-PP_JITTER_FRAC * hex_side,
                                   PP_JITTER_FRAC * hex_side)
            jy = vy + rng.uniform(-PP_JITTER_FRAC * hex_side,
                                   PP_JITTER_FRAC * hex_side)
            if 0 <= int(jx) < w and 0 <= int(jy) < h and tissue[int(jy), int(jx)]:
                rx = rng.integers(PP_RX[0], PP_RX[1] + 1)
                ry = rng.integers(PP_RY[0], PP_RY[1] + 1)
                paint_blob_mask(portal_mask, jx, jy, rx, ry, rng)

    central_mask &= tissue
    portal_mask &= tissue
    return central_mask, portal_mask, np.array(gt_centers_out, dtype=np.float64)


def _find_label_junctions(labels, kept_ids):
    """Find approximate junction points where 3+ lobule labels meet."""
    h, w = labels.shape
    junctions = []

    # Downsample for speed
    scale = 4
    small = cv2.resize(labels.astype(np.float32), (w // scale, h // scale),
                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
    sh, sw = small.shape

    for y in range(1, sh - 1):
        for x in range(1, sw - 1):
            patch = small[y - 1:y + 2, x - 1:x + 2]
            unique = set(patch.ravel())
            unique.discard(0)
            if len(unique) >= 3:
                junctions.append((x * scale, y * scale))

    # Cluster nearby junctions (merge within hex_side/4)
    if not junctions:
        return []
    pts = np.array(junctions, dtype=np.float64)
    merged = []
    used = np.zeros(len(pts), dtype=bool)
    merge_dist = 40  # pixels
    for i in range(len(pts)):
        if used[i]:
            continue
        cluster = [pts[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            if np.linalg.norm(pts[i] - pts[j]) < merge_dist:
                cluster.append(pts[j])
                used[j] = True
        merged.append(np.mean(cluster, axis=0))
    return merged


# ---------------------------------------------------------------------------
#  Portality computation
# ---------------------------------------------------------------------------
def compute_portality(labels, central_mask, portal_mask, kept_ids):
    """Compute portality per lobule: 0 at boundary, 1 at CV, NaN outside.

    Delegates to :func:`slidekick.processing.lobule_segmentation.portality.lobule_portality`.
    ``kept_ids`` is accepted for backwards compatibility but ignored
    (``lobule_portality`` iterates all non-zero labels automatically).

    The tissue boundary is treated as a lobule boundary (portality 0), matching
    real-image behaviour where the tissue edge functions as the outermost lobule
    wall.
    """
    from slidekick.processing.lobule_segmentation.portality import lobule_portality
    return lobule_portality(labels, central_mask, portal_mask)


def compute_pp_portality(labels, central_mask, portal_mask, kept_ids):
    """PP-specific portality: 0 at portal vessels, 1 at CV, NaN outside.

    Unlike the regular portality (which is 0 at ALL lobule boundaries),
    this is 0 only where portal vessels are. The PP stain derived from this
    is therefore high only near portal vessels - it does NOT appear at tissue
    edges, tissue cutouts, or at shared lobule boundaries that have no portal
    vessel. This eliminates the visual "adding" artefact at non-portal edges.

    Fallback when a lobule has no portal vessel: use its boundary with other
    labeled lobules (not the tissue/background edge).
    """
    h, w = labels.shape
    portality = np.full((h, w), np.nan, dtype=np.float32)

    for label_id in kept_ids:
        region = labels == label_id
        if region.sum() < 10:
            continue

        # PP source: portal vessels within this lobule
        pp_src = portal_mask & region

        if not np.any(pp_src):
            # Fallback: shared boundary with OTHER labeled lobules only
            # (deliberately exclude label==0 / tissue edge / holes)
            fallback = np.zeros((h, w), dtype=bool)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = np.roll(np.roll(labels, -dy, 0), -dx, 1)
                fallback |= region & (nb != label_id) & (nb > 0)
            if np.any(fallback):
                pp_src = fallback
            else:
                # Last resort: any non-self pixel (shouldn't normally happen)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = np.roll(np.roll(labels, -dy, 0), -dx, 1)
                    fallback |= region & (nb != label_id)
                pp_src = fallback

        # CV within this lobule; centroid proxy if absent
        cv_in = central_mask & region
        if not np.any(cv_in):
            ys, xs = np.where(region)
            cy, cx = int(ys.mean()), int(xs.mean())
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < h and 0 <= xx < w and region[yy, xx]:
                        cv_in[yy, xx] = True

        d_pp = edt(~pp_src)
        d_cv = edt(~cv_in)
        denom = d_pp + d_cv + 1e-8
        P = (d_pp / denom).astype(np.float32)
        portality[region] = P[region]

    portality[portal_mask & ~np.isnan(portality)] = 0.0
    portality[central_mask & ~np.isnan(portality)] = 1.0
    return portality


# ---------------------------------------------------------------------------
#  Lobule boundary fusion
# ---------------------------------------------------------------------------
def apply_boundary_fusion(labels, kept_ids, portal_mask, hex_side, rng,
                           threshold=FUSION_THRESHOLD,
                           v_power=FUSION_V_POWER):
    """For each Voronoi edge between adjacent lobules, sample v = U^(1/v_power).

    v < threshold  -> edge ABSENT from GT: lobules merge into one super-lobule.
                     Portal vessels that become interior are removed.
                     Portality is NOT modified here - caller should recompute
                     compute_portality() on the merged labels so the fused edge
                     becomes an interior region and portality flows naturally.
    v >= threshold -> edge PRESENT in GT: no change.

    Returns
    -------
    labels               : (H,W) int32, merged - fused lobules share smallest ID
    kept_ids             : list, updated to only contain root super-lobule IDs
    portal_mask          : (H,W) bool, interior vessels removed
    fused_edges          : list of (id_a, id_b, v_e) for fused pairs
    fused_boundary_mask  : (H,W) bool, pixels on either side of every fused edge
    """
    labels      = labels.copy()
    portal_mask = portal_mask.copy()
    h, w  = labels.shape
    kept  = set(kept_ids)

    # --- 1. Fast lookup table: is a label in kept_ids? ---
    max_id = int(labels.max()) + 1
    in_kept = np.zeros(max_id + 1, bool)
    for kid in kept:
        if kid <= max_id:
            in_kept[kid] = True

    # --- 2. Find all adjacent lobule pairs (horizontal + vertical neighbours) ---
    adjacent = set()
    for a_sl, b_sl in [
        (labels[:-1, :], labels[1:, :]),   # vertical neighbours
        (labels[:, :-1], labels[:, 1:]),   # horizontal neighbours
    ]:
        diff = (a_sl != b_sl)
        a_ok = in_kept[np.clip(a_sl, 0, max_id)]
        b_ok = in_kept[np.clip(b_sl, 0, max_id)]
        sel  = diff & a_ok & b_ok
        for ai, bi in zip(a_sl[sel].tolist(), b_sl[sel].tolist()):
            adjacent.add((min(ai, bi), max(ai, bi)))

    # --- 3. Sample v for each pair; classify as fused or not ---
    # v = U^(1/power): skewed toward 1. P(v < t) = t^power.
    # Each lobule may participate in AT MOST ONE fusion (no chains A-B-C).
    # Sort candidates by v ascending so the "most fused" edges get first pick,
    # then skip any pair where either lobule is already committed.
    candidates = []
    for id_a, id_b in sorted(adjacent):   # sorted for reproducibility
        u   = float(rng.uniform(0.0, 1.0))
        v_e = u ** (1.0 / v_power)
        if v_e < threshold:
            candidates.append((v_e, id_a, id_b))

    candidates.sort()                      # ascending v_e: strongest fusions first
    already_fused: set = set()
    fused_edges = []
    for v_e, id_a, id_b in candidates:
        if id_a in already_fused or id_b in already_fused:
            continue                        # one of them is already in a super-lobule
        fused_edges.append((id_a, id_b, v_e))
        already_fused.add(id_a)
        already_fused.add(id_b)

    # --- 3b. Pixel mask of fused boundaries (BEFORE label merging) ---
    # Marks every pixel on either side of a fused Voronoi edge.
    # Returned so the caller can suppress PP stain there:
    # no portal tract at a fused boundary -> PP signal should be absent.
    fused_boundary_mask = np.zeros((h, w), dtype=bool)
    for id_a, id_b, _ in fused_edges:
        for dy, dx in ((0, 1), (1, 0)):
            la = labels[:h - dy if dy else h, :w - dx if dx else w]
            lb = labels[dy:, dx:]
            bnd = ((la == id_a) & (lb == id_b)) | ((la == id_b) & (lb == id_a))
            fused_boundary_mask[:h - dy if dy else h, :w - dx if dx else w] |= bnd
            fused_boundary_mask[dy:, dx:] |= bnd

    # --- 4. Merge GT labels via union-find ---
    parent = {i: i for i in kept}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        px, py = _find(x), _find(y)
        if px != py:
            # keep smaller ID as canonical root
            if px > py:
                px, py = py, px
            parent[py] = px

    for id_a, id_b, _ in fused_edges:
        if id_a in parent and id_b in parent:
            _union(id_a, id_b)

    # Remap every non-root id -> its root
    root_map = {i: _find(i) for i in kept}
    old_labels = labels.copy()
    for old_id, new_id in root_map.items():
        if old_id != new_id:
            labels[old_labels == old_id] = new_id

    # New kept_ids = roots only
    new_kept = sorted({_find(i) for i in kept})

    # --- 6. Remove portal vessels interior to merged super-lobules ---
    # A portal vessel pixel is interior if ALL positive-label neighbours
    # share the same super-lobule ID.
    pm_ys, pm_xs = np.where(portal_mask)
    if len(pm_ys) > 0:
        # Neighbour labels at portal pixels for 4 directions
        neigh = np.stack([
            np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ], axis=0)                          # (4, H, W)
        nl = neigh[:, pm_ys, pm_xs]        # (4, n_portal)
        pos = nl > 0                        # positive (lobule) neighbours
        has_pos = pos.any(axis=0)
        nl_min = np.where(pos, nl, nl.max() + 1).min(axis=0)
        nl_max = np.where(pos, nl, -1).max(axis=0)
        # Interior: no lobule neighbour, or all lobule neighbours same label
        interior = ~has_pos | (nl_min == nl_max)
        portal_mask[pm_ys[interior], pm_xs[interior]] = False

    return labels, new_kept, portal_mask, fused_edges, fused_boundary_mask


# ---------------------------------------------------------------------------
#  Stain generation
# ---------------------------------------------------------------------------
def _make_noise(shape, rng, sigma_spatial=8.0):
    """Spatially correlated Gaussian noise in [0, 1]."""
    raw = rng.standard_normal(shape).astype(np.float32)
    smooth = gaussian_filter(raw, sigma=sigma_spatial)
    smooth -= smooth.min()
    mx = smooth.max()
    if mx > 0:
        smooth /= mx
    return smooth


def _make_shading_field(shape, rng, lo: float = 0.65, hi: float = 1.0) -> np.ndarray:
    """Smooth multiplicative flatfield-degradation field in [lo, hi].

    Simulates uneven staining intensity across the slide - e.g. an antibody
    concentration gradient from one edge to the other, or a slow wash-out
    pattern.  A single broad Gaussian-smoothed noise field is used so the
    variation is image-wide and correlated at the lobule scale.

    Sigma is drawn from U[30 %, 60 %] of the image width so the field is
    always broader than a full lobule diameter (~560 px at hex_side=280 px /
    1600 px image = 35 % of width).  This ensures the shading gradient is
    always separable from the within-lobule zonation gradient, so the
    flatfield correction (local_bg_sigma ≥ 400 px) can cleanly remove it
    without destroying the PV/PP signal.

    Parameters
    ----------
    shape : (H, W)
    lo, hi : float  - multiplier range; default [0.50, 1.0]

    Returns
    -------
    field : (H, W) float32  in [lo, hi]
    """
    h, w = shape
    sigma = float(rng.uniform(0.30 * max(h, w), 0.60 * max(h, w)))
    raw   = rng.standard_normal((h, w)).astype(np.float32)
    field = gaussian_filter(raw, sigma=sigma)
    f_min, f_max = float(field.min()), float(field.max())
    if f_max > f_min:
        field = (field - f_min) / (f_max - f_min)   # -> [0, 1]
    else:
        field = np.ones_like(field)
    return (lo + field * (hi - lo)).astype(np.float32)


def _make_illumination_field(shape, rng, i_min=0.4):
    """Generate a smooth multiplicative illumination field in [i_min, 1.0].

    Models real microscope non-uniformity:
      - Low-freq component  (sigma~500-800px): vignetting / excitation field
      - Mid-freq component  (sigma~150-300px): tissue thickness / section quality

    Both are combined into a single multiplicative field.
    """
    h, w = shape

    # Low-frequency: large-scale illumination variation
    sigma_low = rng.uniform(400, 800)
    low = rng.standard_normal((h, w)).astype(np.float32)
    low = gaussian_filter(low, sigma=sigma_low)

    # Mid-frequency: tissue patch variation
    sigma_mid = rng.uniform(100, 300)
    mid = rng.standard_normal((h, w)).astype(np.float32)
    mid = gaussian_filter(mid, sigma=sigma_mid)

    # Combine (equal weight)
    field = 0.6 * low + 0.4 * mid
    # Normalize to [0, 1]
    field -= field.min()
    mx = field.max()
    if mx > 0:
        field /= mx
    # Rescale to [i_min, 1.0]
    field = i_min + field * (1.0 - i_min)
    return field.astype(np.float32)


def _make_dapi_ecad_texture(shape, tissue, rng,
                             target_mean=70.0, target_std=35.0,
                             nuclear_radius_px=3.0):
    """High-frequency structural noise channel matching the combined
    nuclear + membrane channel appearance of real multiplexed fluorescence images.

    Real measurements (averaged across 7 images, inside tissue, uint8):
        mean ≈ 70, std ≈ 35,
        local 9x9 std ≈ 27 (so pixel-to-pixel variation is most of total std).

    The texture is built as a sum of:
      1. Nuclear dots - sparse Gaussian blobs at a density ≈ 1 per
         (2*nuclear_radius)^2 px², contributing bright spots for DAPI.
      2. Medium-frequency background (blurred Gaussian noise, σ~6 px)
         contributing cytoplasm/E-Cad membrane patterns.
      3. A small low-frequency illumination gradient.

    Returns a uint8 (H,W) array.
    """
    h, w = shape
    # 1) Nuclear dots (DAPI)
    nuclei = np.zeros((h, w), dtype=np.float32)
    area = h * w
    # ~one nucleus per 30 px² -> very dense in tissue, close to real
    n_nuclei = int(area / 30)
    ys = rng.integers(0, h, n_nuclei)
    xs = rng.integers(0, w, n_nuclei)
    # Only place nuclei on tissue
    keep = tissue[ys, xs]
    ys, xs = ys[keep], xs[keep]
    # Random brightness per nucleus
    amps = rng.uniform(0.4, 1.0, size=len(ys)).astype(np.float32)
    nuclei[ys, xs] = amps
    # Blur to nuclear size
    nuclei = gaussian_filter(nuclei, sigma=nuclear_radius_px)
    # Normalize so peak ≈ 1
    mx = nuclei.max()
    if mx > 0:
        nuclei /= mx

    # 2) Medium-frequency background (E-Cadherin membranes + cytoplasm)
    bg = rng.standard_normal((h, w)).astype(np.float32)
    bg = gaussian_filter(bg, sigma=6.0)
    bg -= bg.mean()
    s = bg.std()
    if s > 0:
        bg /= s

    # 3) Low-frequency illumination
    illum = rng.standard_normal((h, w)).astype(np.float32)
    illum = gaussian_filter(illum, sigma=80.0)
    illum -= illum.mean()
    s = illum.std()
    if s > 0:
        illum /= s

    # Combine - nuclei dominate the bright peaks, bg provides the mid tones
    combined = 0.75 * nuclei + 0.40 * bg + 0.10 * illum

    # Rescale to target mean/std inside tissue
    m = combined[tissue]
    if m.size > 0 and m.std() > 1e-6:
        combined = (combined - m.mean()) / m.std()
        combined = combined * target_std + target_mean
    else:
        combined = np.full_like(combined, target_mean)

    # Clip
    combined = np.clip(combined, 0, 255)
    # Zero outside tissue
    out = np.zeros_like(combined, dtype=np.uint8)
    out[tissue] = combined[tissue].astype(np.uint8)
    return out


def _match_channel_stats(stain01, tissue, vessel_holes, target_p95, target_p05=10.0):
    """Rescale a float stain to uint8 so the in-tissue stats match targets.

    Uses a p05->p99 anchor pair (rather than p05->p95) so the bright tail of
    the source has headroom in uint8 space instead of saturating at 255.

    - source p05 -> target_p05
    - source p99 -> target_p99  (derived from target_p95 by a fixed ratio)
    - values above p99 extend smoothly toward 255 (no hard clip until max)

    The real PV marker has p95/p99 ≈ 0.77, so target_p99 ≈ target_p95 / 0.77.
    """
    fg = tissue & ~vessel_holes
    vals = stain01[fg]
    if vals.size == 0:
        return np.zeros_like(stain01, dtype=np.uint8)
    src_lo = float(np.percentile(vals, 5))
    src_hi = float(np.percentile(vals, 99))
    if src_hi <= src_lo:
        return np.zeros_like(stain01, dtype=np.uint8)

    # Real p95/p99 ratio for the PV marker in calibration is ~0.77, giving
    # target_p99 ≈ target_p95 / 0.77. Cap at 245 to leave room for max.
    target_p99 = min(245.0, target_p95 / 0.77)
    scale = (target_p99 - target_p05) / (src_hi - src_lo)
    offset = target_p05 - scale * src_lo
    out = stain01 * scale + offset
    out = np.clip(out, 0, 255)
    out[~fg] = 0
    return out.astype(np.uint8)


def _make_cellular_texture(shape, tissue, rng, cell_sigma=2.0, cell_amp=0.25):
    """Generate cellular granular texture mimicking individual cell fluorescence.

    Real fluorescence images show rapid pixel-to-pixel intensity variation
    from individual cells. We model this as fine-grained multiplicative noise
    at the cell scale (~2-4 px at typical pyramid levels).

    Returns a multiplicative field centered on 1.0 with std ~ cell_amp.
    """
    h, w = shape
    # Fine noise at cell scale
    raw = rng.standard_normal((h, w)).astype(np.float32)
    # Slight blur to mimic cell size (not single-pixel)
    raw = gaussian_filter(raw, sigma=cell_sigma)
    # Centre on 1.0 with controlled amplitude
    raw = raw / (raw.std() + 1e-8) * cell_amp
    texture = 1.0 + raw
    texture = np.clip(texture, 0.2, 2.0)
    texture[~tissue] = 1.0
    return texture.astype(np.float32)


def _make_lognormal_cell_texture(shape, tissue, rng,
                                   sigma=_CELL_LOGNORMAL_SIGMA,
                                   corr_sigma=_CELL_LOGNORMAL_CORR):
    """Log-normal cellular multiplier with heavy bright tail.

    mult = exp(sigma * Z) where Z is a unit-std Gaussian field smoothed with
    correlation length ``corr_sigma`` (in pixels). Median ≈ 1, mean ≈
    exp(sigma**2/2). With sigma=0.45: ~5% of pixels exceed 2.1x and ~5% are
    below 0.48x. Re-centred so the in-tissue mean is 1.0 so downstream
    _match_channel_stats still behaves.
    """
    h, w = shape
    z = rng.standard_normal((h, w)).astype(np.float32)
    z = gaussian_filter(z, sigma=corr_sigma)
    # Renormalize to unit std
    z = z / (z.std() + 1e-8)
    mult = np.exp(sigma * z).astype(np.float32)
    # Re-centre mean to 1.0 inside tissue so the global brightness isn't biased
    fg = tissue
    if fg.any():
        mean_in = mult[fg].mean()
        if mean_in > 0:
            mult /= mean_in
    mult[~tissue] = 1.0
    return mult


def generate_stain(portality, tissue, vessel_holes, rng,
                   stain_type="linear", noise_level="low",
                   channel="pv",
                   lobule_expr_map=None, fold_map=None,
                   lobule_base_map=None, lobule_k_map=None,
                   lobule_amp_map=None,
                   cv_offset_map=None):
    """Generate a stain image from portality.

    Parameters
    ----------
    portality : (H,W) float32, 0=boundary, 1=CV, NaN=outside
    channel : "pv" (pericentral, high near CV) or "pp" (periportal, high near boundary)
    stain_type : "linear" (smooth gradient) | "sharp" (exponential) |
                 "hill" (sigmoidal - matches real PV marker best, R²≈0.95)
    noise_level : "low" or "high"
    lobule_expr_map : (H,W) float32, per-lobule expression multiplier
    lobule_base_map : (H,W) float32, per-lobule PV baseline (defaults to
                      calibrated PP_BASELINE everywhere if None)
    lobule_k_map    : (H,W) float32, per-lobule PV decay rate (defaults to
                      calibrated k_pv everywhere if None)
    fold_map : (H,W) float32, tissue fold intensity
    """
    h, w = portality.shape
    P = portality.copy()
    valid = ~np.isnan(P)

    # -- CV-offset jitter (PV only) --------------------------------
    # Real PV intensity at CV reaches only ~60% of per-lobule p99 because the
    # geometric CV and the actual CYP hotspot don't coincide. Shift the
    # effective portality per lobule, plus a smooth low-freq spatial field,
    # so the peak is smeared off the CV.
    if channel == "pv":
        if cv_offset_map is not None:
            off = cv_offset_map.astype(np.float32)
        else:
            off = np.full_like(P, 0.5 * sum(_CV_OFFSET_RANGE), dtype=np.float32)
        # Low-freq spatial noise, centered in [-amp, +amp].
        _raw = rng.standard_normal((h, w)).astype(np.float32)
        _sm = gaussian_filter(_raw, sigma=_CV_SMOOTH_SIGMA)
        _sm -= _sm.mean()
        _smax = max(abs(_sm.min()), abs(_sm.max()), 1e-6)
        _sm = (_sm / _smax) * _CV_SMOOTH_AMP
        # Apply jitter only where portality is defined; leave NaNs alone.
        P_jit = P.copy()
        P_jit[valid] = np.clip(P[valid] - off[valid] + _sm[valid], 0.0, 1.0)
        P = P_jit

    # -- Per-pixel base/k (defaults constant from calibration) --
    default_base = float(PP_BASELINE_CAL) if PP_BASELINE_CAL is not None else 0.30
    if lobule_base_map is not None:
        base_field = lobule_base_map.astype(np.float32)
    else:
        base_field = np.full_like(P, default_base, dtype=np.float32)

    default_k = float(np.log(2) / max(PV_SIGMA_FRAC, 1e-6))
    if lobule_k_map is not None:
        k_field = lobule_k_map.astype(np.float32)
    else:
        k_field = np.full_like(P, default_k, dtype=np.float32)

    # -- Portality -> normalized shape in [base, 1] --
    intensity = np.zeros_like(P, dtype=np.float32)

    if channel == "pv":
        if stain_type == "linear":
            shape = P
        elif stain_type == "sharp":
            # exponential rise toward CV (p=1)
            # f(p) = exp(-k*(1-p)) normalized to [0,1] per lobule
            raw = np.exp(-k_field * (1.0 - np.clip(P, 0, 1)))
            # normalize per-lobule to [0,1]: here use the global min (1 at p=0 -> exp(-k))
            # but since we add base_field below, just use raw - exp(-k)  over 1-exp(-k)
            fmin = np.exp(-k_field)
            shape = (raw - fmin) / (1.0 - fmin + 1e-12)
        else:  # "hill" - sigmoidal, measured to fit R²≈0.95 on real PV marker
            n_hill, h_hill = float(_CAL_HILL_PV[1]), float(_CAL_HILL_PV[2])
            u = np.clip(P, 0, 1)
            num = u ** n_hill
            den = num + h_hill ** n_hill + 1e-12
            shape = num / den  # in [0, 1/(1+h^n)]
            # Renormalize so shape reaches 1 at p=1
            shape = shape / (1.0 / (1.0 + h_hill ** n_hill) + 1e-12)
    else:  # pp
        # PP marker is highest near lobule boundaries and decays toward the CV.
        # Use a portality-based exponential so PP rings ALL lobule edges.
        # Steeper k_pp (half-max at P≈0.15) keeps the ring thin so adjacent
        # lobules' PP zones don't visually merge.
        if stain_type == "linear":
            shape = 1.0 - P
        else:
            k_pp = np.log(2) / 0.65   # half-max at portality = 0.65 (very wide, low contrast)
            raw  = np.exp(-k_pp * np.clip(P, 0, 1))
            shape = (raw - np.exp(-k_pp)) / (1.0 - np.exp(-k_pp) + 1e-12)

    # Per-lobule amplitude cap: mean curve peaks at `amp` (not 1); the
    # remaining headroom is filled in later by the log-normal cell texture.
    if channel == "pv" and lobule_amp_map is not None:
        amp_field = lobule_amp_map.astype(np.float32)
    else:
        amp_field = np.ones_like(P, dtype=np.float32)

    # Decompose intensity into two components that are modulated separately:
    #   base_contrib  - autofluorescence-like floor, unaffected by CYP
    #                   expression / cellular clustering, but still touched
    #                   by microscope illumination
    #   shape_contrib - CYP signal proportional to the stain shape; modulated
    #                   by lobule expression, illum, and (for PV) the
    #                   log-normal cellular multiplier that produces bright
    #                   CYP cell clusters.
    base_contrib = np.zeros_like(P, dtype=np.float32)
    shape_contrib = np.zeros_like(P, dtype=np.float32)
    base_contrib[valid] = base_field[valid]
    shape_contrib[valid] = (amp_field[valid] - base_field[valid]) * shape[valid]

    m = valid & ~vessel_holes

    # -- Per-lobule CYP expression (affects shape contribution only) --
    if lobule_expr_map is not None and channel == "pv":
        shape_contrib[m] *= lobule_expr_map[m]
    elif lobule_expr_map is not None:
        # non-PV: expression affects everything (PP marker is weakly zonated
        # anyway, keep legacy behaviour)
        shape_contrib[m] *= lobule_expr_map[m]
        base_contrib[m] *= lobule_expr_map[m]

    # -- Multiplicative illumination non-uniformity (affects both) --
    if ENABLE_ILLUMINATION:
        illum_min = 0.5 if noise_level == "low" else 0.3
        illum_field = _make_illumination_field((h, w), rng, i_min=illum_min)
        base_contrib[m] *= illum_field[m]
        shape_contrib[m] *= illum_field[m]

    # -- Cellular granular texture (log-normal for PV, shape-only) --
    if channel == "pv":
        sigma_ln = (_CELL_LOGNORMAL_SIGMA if noise_level == "low"
                    else _CELL_LOGNORMAL_SIGMA * 1.2)
        cell_texture = _make_lognormal_cell_texture(
            (h, w), tissue, rng,
            sigma=sigma_ln,
            corr_sigma=_CELL_LOGNORMAL_CORR)
        shape_contrib[m] *= cell_texture[m]
    else:
        cell_amp = 0.20 if noise_level == "low" else 0.35
        cell_texture = _make_cellular_texture((h, w), tissue, rng,
                                              cell_sigma=2.0, cell_amp=cell_amp)
        # For PP, multiplicative on whole intensity (legacy)
        base_contrib[m] *= cell_texture[m]
        shape_contrib[m] *= cell_texture[m]

    # Recompose intensity
    intensity[valid] = base_contrib[valid] + shape_contrib[valid]
    intensity[~valid] = 0.0
    intensity[vessel_holes] = 0.0

    # -- Additive smooth noise (original) --
    noise_amp = 0.10 if noise_level == "low" else 0.35
    noise = _make_noise((h, w), rng)
    intensity[valid & ~vessel_holes] += noise_amp * noise[valid & ~vessel_holes]

    # -- Autofluorescence baseline --
    # Real tissue has non-zero signal even in periportal "dark" regions
    intensity[valid & ~vessel_holes] += AUTOFLUO_FRAC

    intensity = np.clip(intensity, 0.0, None)

    # -- Tissue fold artifact --
    if fold_map is not None:
        intensity[valid & ~vessel_holes] += fold_map[valid & ~vessel_holes]
        intensity = np.clip(intensity, 0.0, None)

    # -- Global brightness variability --
    # Real images differ 2-3x in overall brightness
    global_mult = rng.uniform(GLOBAL_BRIGHTNESS_MIN, GLOBAL_BRIGHTNESS_MAX)
    intensity[valid & ~vessel_holes] *= global_mult

    # Normalize to [0, 1]
    mx = intensity.max()
    if mx > 0:
        intensity /= mx

    return intensity.astype(np.float32)


def _add_scanbox(tissue, stain_u8, vessel_holes):
    """Add scan-background gray around tissue."""
    h, w = tissue.shape
    ys, xs = np.where(tissue)
    y0 = max(0, ys.min() - SCAN_MARGIN)
    y1 = min(h - 1, ys.max() + SCAN_MARGIN)
    x0 = max(0, xs.min() - SCAN_MARGIN)
    x1 = min(w - 1, xs.max() + SCAN_MARGIN)
    out = np.zeros((h, w), dtype=np.uint8)
    out[y0:y1 + 1, x0:x1 + 1] = SCAN_GRAY
    out = np.maximum(out, stain_u8)
    out[vessel_holes] = 0
    return out


# ---------------------------------------------------------------------------
#  Main instance generator
# ---------------------------------------------------------------------------
def _generate_geometry(seed, hex_side, verbose: bool = True):
    """Generate tissue, lobule labels, vessels, portality for one seed+hex_side.

    Returns a dict of geometry arrays (shared across stain variants).
    """
    rng = np.random.default_rng(seed)

    # Tissue boundary: generated procedurally from measured EFD statistics.
    tissue = tissue_from_outline(IMG, rng)
    centers = hex_grid_centers(IMG, hex_side, rng)

    labels, kept_ids = build_perturbed_voronoi_labels(
        centers, tissue, hex_side, rng)

    central_mask, portal_mask, gt_centers = place_vessels(
        centers, kept_ids, labels, tissue, hex_side, rng)

    vessel_holes = central_mask | portal_mask

    # Compute portality BEFORE zeroing vessel holes from labels,
    # because portality needs the full lobule region including CV positions
    portality = compute_portality(labels, central_mask, portal_mask, kept_ids)
    pp_portality = compute_pp_portality(labels, central_mask, portal_mask, kept_ids)

    def _smooth_pp_portality(pp_port, hex_s):
        """Tissue-aware Gaussian blur to soften portal-vessel PP anchoring."""
        valid = ~np.isnan(pp_port)
        sigma = max(18.0, 0.13 * hex_s)
        tmp = pp_port.copy()
        tmp[~valid] = 0.5          # neutral fill to avoid edge bleeding
        blurred = gaussian_filter(tmp.astype(np.float32), sigma=sigma)
        result = pp_port.copy()
        result[valid] = np.clip(blurred[valid], 0.0, 1.0)
        return result

    pp_portality = _smooth_pp_portality(pp_portality, hex_side)

    # Randomly fuse adjacent lobule pairs:
    #   v < FUSION_THRESHOLD -> merge in GT, remove interior portal vessels
    #   v >= FUSION_THRESHOLD -> keep separate, full dark valley
    _n_orig = len(kept_ids)
    labels, kept_ids, portal_mask, fused_pairs, fused_boundary_mask = apply_boundary_fusion(
        labels, kept_ids, portal_mask, hex_side, rng)
    vessel_holes = central_mask | portal_mask   # recompute after portal removal

    # Recompute portality on merged labels so fused edges become interior:
    # portality flows naturally across them (two CVs, no forced zero at shared edge).
    _n_fused  = len(fused_pairs)
    _n_final  = len(kept_ids)
    _n_super  = _n_fused          # each fused edge creates one super-lobule from two originals
    _n_unchanged = _n_final - _n_super
    if fused_pairs:
        portality = compute_portality(labels, central_mask, portal_mask, kept_ids)
        pp_portality = _smooth_pp_portality(
            compute_pp_portality(labels, central_mask, portal_mask, kept_ids), hex_side)

        # Suppress PP stain at fused edges: a fused boundary has no portal tract
        # -> the PP marker should show no ring there.  Blend pp_portality toward
        # 1.0 (= "far from portal" -> PP shape ≈ 0) over a Gaussian zone centred
        # on the boundary pixels.  sigma ≈ 15 % of hex_side ≈ half a portal-ring width.
        if fused_boundary_mask.any():
            suppress_sigma = max(8.0, hex_side * 0.15)
            fused_weight = gaussian_filter(
                fused_boundary_mask.astype(np.float32), sigma=suppress_sigma)
            fused_max = float(fused_weight.max())
            if fused_max > 0:
                fused_weight = np.clip(fused_weight / fused_max, 0.0, 1.0)
            # Apply only inside tissue (NaN pixels stay NaN).
            valid_pp = ~np.isnan(pp_portality)
            pp_portality[valid_pp] = (
                pp_portality[valid_pp] * (1.0 - fused_weight[valid_pp])
                + 1.0 * fused_weight[valid_pp]
            )

        if verbose:
            print(f"    {_n_orig} lobules | {_n_fused} edge(s) fused → "
                  f"{_n_unchanged} unchanged + {_n_super} super-lobule(s)", flush=True)
    else:
        if verbose:
            print(f"    {_n_orig} lobules | no fusions", flush=True)

    # Now zero out vessel pixels from labels (they're holes, not lobule tissue)
    labels[vessel_holes] = 0

    # Post-vessel fragment absorption:
    # Vessel holes can punch through a lobule edge and disconnect a corner
    # sliver from its main body.  These slivers are NOT independent lobules -
    # absorb each one into its dominant neighbour using the Voronoi topology.
    # Uses 8-connected components; threshold = 10 % of a full hex cell area.
    _frag_thr = int(0.10 * hex_side ** 2)
    _conn8    = _ndstruct(2, 2)               # 8-connectivity structuring element
    _cc_map, _n_cc = _ndlabel(labels > 0, structure=_conn8)
    if _n_cc > 0:
        _sizes = np.bincount(_cc_map.ravel())   # index 0 = background
        for _cc_id in range(1, _n_cc + 1):
            if _sizes[_cc_id] >= _frag_thr:
                continue
            _frag = _cc_map == _cc_id
            # 1-px dilation to find adjacent non-zero pixels outside the fragment
            _nbr_mask = _nddilate(_frag, structure=_conn8) & (labels > 0) & ~_frag
            if _nbr_mask.any():
                _nbr_vals, _counts = np.unique(labels[_nbr_mask], return_counts=True)
                labels[_frag] = _nbr_vals[np.argmax(_counts)]   # dominant neighbour
            else:
                labels[_frag] = 0   # isolated island - treat as background

    # Per-lobule expression heterogeneity (calibrated range)
    lobule_expr_map = generate_lobule_expression_map(labels, kept_ids, rng)
    # Per-lobule PV shape heterogeneity - each lobule gets its own
    # (base, k) drawn from the measured per-lobule distribution, so the
    # synthetic per-lobule stat distribution matches the real one.
    lobule_base_map = generate_lobule_base_map(labels, kept_ids, rng)
    lobule_k_map = generate_lobule_k_map(labels, kept_ids, rng)
    # Per-lobule PV amplitude cap (A) - mean stain curve peaks at A, not 1
    lobule_amp_map = generate_lobule_amp_map(labels, kept_ids, rng)
    # Per-lobule CV-offset (peak-away-from-CV jitter, see generate_stain)
    cv_offset_map = generate_cv_offset_map(labels, kept_ids, rng)

    # Smooth the hard per-lobule boundaries so neighbouring lobules blend
    # gradually instead of stepping abruptly. Tissue-aware Gaussian with
    # sigma = 18% of the hex side gives a soft transition ~1/3 of a lobule
    # wide without averaging away the per-lobule character.
    smooth_sigma = max(4.0, 0.18 * hex_side)
    lobule_expr_map = _smooth_lobule_map(
        lobule_expr_map, tissue, smooth_sigma,
        fill=float(0.5 * (LOBULE_EXPR_MIN + LOBULE_EXPR_MAX)))

    lobule_base_map = _smooth_lobule_map(
        lobule_base_map, tissue, smooth_sigma,
        fill=float(0.5 * (_CAL_PV_BASE_RANGE[0] + _CAL_PV_BASE_RANGE[1])))
    lobule_k_map = _smooth_lobule_map(
        lobule_k_map, tissue, smooth_sigma,
        fill=float(0.5 * (_CAL_PV_K_RANGE[0] + _CAL_PV_K_RANGE[1])))
    lobule_amp_map = _smooth_lobule_map(
        lobule_amp_map, tissue, smooth_sigma,
        fill=float(0.5 * (_CAL_PV_AMP_RANGE[0] + _CAL_PV_AMP_RANGE[1])))

    # Tissue fold artifact (stochastic - only some images get folds)
    fold_map = None
    if rng.random() < FOLD_PROB:
        fold_map = generate_tissue_fold((IMG, IMG), tissue, rng)

    return dict(
        tissue=tissue, labels=labels, kept_ids=kept_ids,
        central_mask=central_mask, portal_mask=portal_mask,
        vessel_holes=vessel_holes, gt_centers=gt_centers,
        portality=portality, pp_portality=pp_portality,
        fused_pairs=fused_pairs, fused_boundary_mask=fused_boundary_mask, rng=rng,
        lobule_expr_map=lobule_expr_map,
        lobule_base_map=lobule_base_map,
        lobule_k_map=lobule_k_map,
        lobule_amp_map=lobule_amp_map,
        cv_offset_map=cv_offset_map,
        fold_map=fold_map,
    )


def generate_all_instances(
    seeds=(42, 123),
    hex_sides=None,           # default derived from GT calibration (p25/p75)
    pv_stain_types=None,      # tuple of stain types for the PV channel
    pp_stain_types=None,      # tuple of stain types for the PP channel
    noise_levels=("low", "high"),
    modes=("dual", "single"),
    verbose=True,
):
    """Generate all combinations of synthetic instances.

    ``pv_stain_types`` and ``pp_stain_types`` are zipped pairwise, so index i
    of pv_stain_types is always paired with index i of pp_stain_types.
    They must have the same length.

    Supported stain type strings: "linear", "sharp", "hill".

    Returns a list of SyntheticInstance objects.
    """
    if hex_sides is None:
        hex_sides = _CAL_HEX_SIDES
    if pv_stain_types is None:
        pv_stain_types = ("hill", "linear")   # hill = sigmoidal, R²=0.95 on real PV marker
    if pp_stain_types is None:
        pp_stain_types = ("linear", "linear")
    if len(pv_stain_types) != len(pp_stain_types):
        raise ValueError(
            f"pv_stain_types and pp_stain_types must have the same length, "
            f"got {len(pv_stain_types)} vs {len(pp_stain_types)}")
    instances = []

    for seed in seeds:
        for hex_side in hex_sides:
            _key = (seed, hex_side)
            if _key in _GEOM_CACHE:
                geom = _GEOM_CACHE[_key]
                if verbose:
                    n_lob = len(geom["kept_ids"])
                    print(f"  Reusing cached geometry: seed={seed}, hex_side={hex_side}"
                          f"  ({n_lob} lobules)", flush=True)
            else:
                if verbose:
                    print(f"  Generating geometry: seed={seed}, hex_side={hex_side}...",
                          end="", flush=True)
                geom = _generate_geometry(seed, hex_side, verbose=verbose)
                _GEOM_CACHE[_key] = geom
                if verbose:
                    n_lob = len(geom["kept_ids"])
                    print(f"  {n_lob} lobules", flush=True)

            for pv_type, pp_type in zip(pv_stain_types, pp_stain_types):
                for noise_level in noise_levels:
                    # Use a sub-seed so stain noise differs per variant
                    stain_rng = np.random.default_rng(
                        seed * 1000 + hash((pv_type, pp_type, noise_level)) % 10000)

                    pv_raw = generate_stain(
                        geom["portality"], geom["tissue"], geom["vessel_holes"],
                        stain_rng, stain_type=pv_type,
                        noise_level=noise_level, channel="pv",
                        lobule_expr_map=geom["lobule_expr_map"],
                        lobule_base_map=geom.get("lobule_base_map"),
                        lobule_k_map=geom.get("lobule_k_map"),
                        cv_offset_map=geom.get("cv_offset_map"),
                        fold_map=geom["fold_map"])

                    # Flatfield / staining-gradient degradation.
                    # A single slow smooth field in [0.50, 1.0] is generated
                    # once per stain variant and applied to BOTH pv_raw and
                    # pp_raw (same field - same slide, same gradient).  It
                    # simulates slide-level illumination/concentration gradients
                    # (independent of the per-pixel microscope illumination
                    # already baked into generate_stain).  The study's
                    # local_bg_sigma parameter is expected to correct for this.
                    _shading = _make_shading_field((IMG, IMG), stain_rng)
                    _fg = geom["tissue"] & ~geom["vessel_holes"]

                    pv_raw = pv_raw * _shading
                    # Re-normalise to [0, 1] using the foreground max so the
                    # relative shape within the tissue is preserved.
                    _fg_max = float(pv_raw[_fg].max()) if _fg.any() else 0.0
                    if _fg_max > 0:
                        pv_raw = pv_raw / _fg_max
                    pv_raw = pv_raw.astype(np.float32)

                    # pp_type="none" -> no PP stain; segmentation must rely on PV alone.
                    # A zero array is used in the image stack so shape stays consistent.
                    if pp_type == "none":
                        pp_raw = None
                        pp_u8  = np.zeros((IMG, IMG), dtype=np.uint8)
                    else:
                        pp_raw = generate_stain(
                            geom["pp_portality"], geom["tissue"], geom["vessel_holes"],
                            stain_rng, stain_type=pp_type,
                            noise_level=noise_level, channel="pp",
                            lobule_expr_map=None,  # no per-lobule brightness variation for PP;
                            # shading field provides global intensity gradient instead
                            fold_map=geom["fold_map"])
                        pp_raw = pp_raw * _shading
                        _pp_fg_max = float(pp_raw[_fg].max()) if _fg.any() else 0.0
                        if _pp_fg_max > 0:
                            pp_raw = pp_raw / _pp_fg_max
                        pp_raw = pp_raw.astype(np.float32)

                    # Match real per-channel p95 (uint8, inside tissue);
                    # ranges from GT-calibration fallbacks.
                    target_p95_cyp1 = float(stain_rng.uniform(*_CAL_CYP1_P95_RANGE))
                    # Real p05 inside annotated lobules is ≈ 2 for both
                    # CYPs. Drop the floor so the dark end of the curve
                    # lines up.
                    pv_u8 = _match_channel_stats(
                        pv_raw, geom["tissue"], geom["vessel_holes"],
                        target_p95=target_p95_cyp1, target_p05=2.0)
                    if pp_raw is not None:
                        target_p95_cyp3 = float(stain_rng.uniform(*_CAL_CYP3_P95_RANGE))
                        pp_u8 = _match_channel_stats(
                            pp_raw, geom["tissue"], geom["vessel_holes"],
                            target_p95=target_p95_cyp3, target_p05=1.0)

                    # Structural noise channel (nuclear + membrane channels merged).
                    # Target mean/std ranges come from GT calibration.
                    noise_u8 = _make_dapi_ecad_texture(
                        (IMG, IMG), geom["tissue"], stain_rng,
                        target_mean=float(stain_rng.uniform(*_CAL_NOISE_MEAN_RANGE)),
                        target_std=float(stain_rng.uniform(*_CAL_NOISE_STD_RANGE)),
                        nuclear_radius_px=float(stain_rng.uniform(2.2, 3.6)))
                    no_u8 = noise_u8

                    if ENABLE_SCANBOX:
                        pv_u8 = _add_scanbox(geom["tissue"], pv_u8, geom["vessel_holes"])
                        if pp_raw is not None:
                            pp_u8 = _add_scanbox(geom["tissue"], pp_u8, geom["vessel_holes"])
                        no_u8 = _add_scanbox(geom["tissue"], no_u8, geom["vessel_holes"])

                    for mode in modes:
                        if mode == "dual":
                            stack = np.stack([pv_u8, pp_u8, no_u8], axis=-1)
                        else:
                            stack = np.stack([pv_u8, no_u8], axis=-1)

                        inst = SyntheticInstance(
                            seed=seed, hex_side=hex_side,
                            stain_type=f"{pv_type}+{pp_type}", noise_level=noise_level,
                            mode=mode,
                            tissue=geom["tissue"],
                            gt_labels=geom["labels"].copy(),
                            gt_portality=geom["portality"].copy(),
                            central_mask=geom["central_mask"],
                            portal_mask=geom["portal_mask"],
                            vessel_holes=geom["vessel_holes"],
                            gt_centers=geom["gt_centers"],
                            image_stack=stack,
                            pv_stain_raw=pv_raw,
                            # pp_stain_raw=None for pp_type="none" AND for "single" mode
                            pp_stain_raw=pp_raw if (mode == "dual" and pp_raw is not None) else None,
                            shading_field=_shading,
                            fused_pairs=geom.get("fused_pairs", []),
                            fused_boundary_mask=geom.get("fused_boundary_mask"),
                        )
                        instances.append(inst)

    if verbose:
        print(f"Generated {len(instances)} instances total.")
    return instances


# ---------------------------------------------------------------------------
#  TIFF export - writes a pyramidal OME-TIFF readable by the full pipeline
# ---------------------------------------------------------------------------
def save_instance_as_tiff(instance, path, pyramid_factors=None):
    """Write a :class:`SyntheticInstance` as a multi-level OME-TIFF.

    Uses :func:`slidekick.io.tif.save_tif` so the output is compatible with
    the annotation tool (``output/annotate_lobules.py``) and the segmentation
    pipeline (``slidekick.processing.lobule_segmentation``).

    Parameters
    ----------
    instance:
        A :class:`SyntheticInstance` returned by :func:`generate_all_instances`.
    path:
        Output ``.tif`` file path.
    pyramid_factors:
        Down-sampling factors for extra pyramid levels, e.g. ``[2, 4]``.
        The full-resolution image is always written as level 0. Defaults to
        ``[2, 4]``.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    from slidekick.io.tif import save_tif
    from slidekick.io.metadata import Metadata

    if pyramid_factors is None:
        pyramid_factors = [2, 4]

    stk  = instance.image_stack  # (H, W, C) uint8
    chw  = np.moveaxis(stk, -1, 0)  # (C, H, W)

    # Build pyramid dict {level_index: array}
    pyramid: dict[int, np.ndarray] = {0: chw}
    for ds in pyramid_factors:
        lvl = int(round(np.log2(ds)))
        pyramid[lvl] = chw[:, ::ds, ::ds]

    # Minimal metadata: channel names so the annotator shows sensible labels
    path = Path(path)
    meta = Metadata(path_original=path, path_storage=path)
    C    = chw.shape[0]
    if instance.mode == "single":
        names = {0: "PV_marker", 1: "noise"}
    else:
        names = {0: "PV_marker", 1: "PP_marker", 2: "noise"}
    meta.stains = {i: names.get(i, f"ch{i}") for i in range(C)}
    meta.ensure_channel_metadata(C)

    save_tif(pyramid, path, metadata=meta)
    return path


# ---------------------------------------------------------------------------
#  Visualization helper - shared by viz_synth.py and the __main__ block
# ---------------------------------------------------------------------------
def visualize_instances(instances, out_path, title_suffix: str = "",
                        thumb_px: int = 512):
    """Render a figure with one row per instance.

    Each row has 4 panels:
        GT labels (coloured) | GT portality | PV stain | PP stain (or grey "no PP")

    Images are downsampled to *thumb_px* on the longer side before display so
    that large canvases (4000 px) don't bloat the output file.

    Parameters
    ----------
    instances : list of SyntheticInstance
    out_path  : str or Path - destination PNG
    title_suffix : str - optional suptitle suffix
    thumb_px  : int - max edge length for each panel thumbnail
    """
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    from pathlib import Path

    import cv2 as _cv2  # local import; cv2 is already a hard dep of the module

    def _thumb(arr, interp=_cv2.INTER_AREA):
        """Downsample *arr* to at most thumb_px on the long edge, preserve dtype."""
        h, w = arr.shape[:2]
        scale = thumb_px / max(h, w)
        if scale >= 1.0:
            return arr
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        if arr.ndim == 2:
            return _cv2.resize(arr, (nw, nh), interpolation=interp)
        # 3-channel float/uint8
        return _cv2.resize(arr, (nw, nh), interpolation=interp)

    cmap_p = plt.get_cmap("magma").copy()
    cmap_p.set_bad("black")
    cmap_s = plt.get_cmap("RdYlGn").copy()  # red=dark shading, green=full intensity
    cmap_s.set_bad("black")

    N_COLS  = 5
    n_inst  = len(instances)
    cell_in = thumb_px / 100        # cell size in inches (100 dpi reference)
    fig, axes = plt.subplots(
        n_inst, N_COLS,
        figsize=(N_COLS * cell_in, n_inst * cell_in),
        squeeze=False,
    )

    for si, inst in enumerate(instances):
        pv_ch = inst.image_stack[..., 0]  # always present
        # PP: present if pp_stain_raw is not None (None for pp_type="none")
        has_pp = inst.pp_stain_raw is not None
        pp_ch  = inst.image_stack[..., 1] if (has_pp and inst.image_stack.shape[-1] >= 2) else None

        # ── Thumbnail all source arrays ───────────────────────────────────────
        pv_t  = _thumb(pv_ch)
        lab_t = _thumb(inst.gt_labels.astype(np.int32), _cv2.INTER_NEAREST)
        por_t = _thumb(inst.gt_portality.astype(np.float32), _cv2.INTER_NEAREST)
        tis_t = _thumb(inst.tissue.astype(np.uint8), _cv2.INTER_NEAREST).astype(bool)

        # ── Fused boundary overlay (thumbnailed, dilated 1px for visibility) ─
        has_fused = (inst.fused_boundary_mask is not None
                     and inst.fused_boundary_mask.any())
        if has_fused:
            fbm_t = _thumb(inst.fused_boundary_mask.astype(np.uint8),
                           _cv2.INTER_NEAREST).astype(bool)
            # Dilate 1 px so the 1-px boundary line is visible at small size
            kern = np.ones((3, 3), np.uint8)
            fbm_t = _cv2.dilate(fbm_t.astype(np.uint8), kern).astype(bool)
        else:
            fbm_t = None

        # ── Panel 0: GT labels (random colour per lobule) ─────────────────────
        n_ids = int(lab_t.max()) + 1
        rng_c = np.random.default_rng(0)
        cols  = rng_c.random((n_ids, 3)).astype(np.float32)
        cols[0] = [0.0, 0.0, 0.0]
        rgb_lab = cols[lab_t]
        rgb_lab[~tis_t] = [0.10, 0.10, 0.10]
        rgb_lab_u8 = (rgb_lab * 255).astype(np.uint8)
        # Overlay fused boundaries as white dashed line
        if fbm_t is not None:
            rgb_lab_u8[fbm_t] = [255, 255, 255]

        n_gt_lob  = len(np.unique(inst.gt_labels[inst.gt_labels > 0]))
        fuse_note = f" [{len(inst.fused_pairs)} fused]" if inst.fused_pairs else ""
        ax = axes[si, 0]
        ax.imshow(rgb_lab_u8)
        ax.set_title(
            f"s{inst.seed} h{inst.hex_side} {inst.stain_type}\n"
            f"GT labels  {n_gt_lob} lobules{fuse_note}",
            fontsize=6, pad=2,
        )

        # ── Panel 1: GT portality ─────────────────────────────────────────────
        ax = axes[si, 1]
        ax.imshow(ma.masked_invalid(por_t), cmap=cmap_p, vmin=0, vmax=1)
        ax.set_title("GT portality", fontsize=6, pad=2)

        # ── Panel 2: PV stain ─────────────────────────────────────────────────
        ax = axes[si, 2]
        ax.imshow(pv_t, cmap="hot", vmin=0, vmax=255)
        p99_pv = float(np.percentile(pv_ch[inst.tissue], 99))
        ax.set_title(f"PV  p99={p99_pv:.0f}", fontsize=6, pad=2)

        # ── Panel 3: PP stain (or grey placeholder when pp_type="none") ───────
        ax = axes[si, 3]
        if pp_ch is not None:
            pp_t   = _thumb(pp_ch)
            p99_pp = float(np.percentile(pp_ch[inst.tissue], 99))
            # Convert hot-colormap image to RGB so we can overlay fused boundary
            pp_norm = np.clip(pp_t.astype(np.float32) / 255.0, 0, 1)
            import matplotlib.cm as _cm
            pp_rgb_f = _cm.hot(pp_norm)[..., :3]          # (H,W,3) float [0,1]
            pp_rgb_u8 = (pp_rgb_f * 255).astype(np.uint8)
            if fbm_t is not None:
                pp_rgb_u8[fbm_t] = [0, 180, 255]          # cyan = suppressed zone
            ax.imshow(pp_rgb_u8)
            fuse_pp_note = "  (cyan=suppressed)" if fbm_t is not None else ""
            ax.set_title(f"PP  p99={p99_pp:.0f}{fuse_pp_note}", fontsize=6, pad=2)
        else:
            # Show a dark grey canvas with centred text
            blank = np.full(pv_t.shape[:2], 30, dtype=np.uint8)
            blank_rgb = np.stack([blank, blank, blank], axis=-1)
            if fbm_t is not None:
                blank_rgb[fbm_t] = [0, 180, 255]          # cyan = where PP would be suppressed
            ax.imshow(blank_rgb)
            ax.text(0.5, 0.5, "no PP", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="white")
            fuse_pp_note = "  (cyan=fused edge)" if fbm_t is not None else ""
            ax.set_title(f"PP  (none){fuse_pp_note}", fontsize=6, pad=2)

        # ── Panel 4: shading field (viridis) ──────────────────────────────────
        ax = axes[si, 4]
        if inst.shading_field is not None:
            shd_t = _thumb(inst.shading_field, _cv2.INTER_AREA)
            _s_min, _s_max = float(shd_t.min()), float(shd_t.max())
            ax.imshow(shd_t, cmap=cmap_s, vmin=_s_min, vmax=_s_max)
            ax.set_title(
                f"shading  [{inst.shading_field.min():.2f}–{inst.shading_field.max():.2f}]",
                fontsize=6, pad=2,
            )
        else:
            blank = np.full(pv_t.shape[:2], 30, dtype=np.uint8)
            ax.imshow(blank, cmap="gray", vmin=0, vmax=255)
            ax.set_title("shading  (none)", fontsize=6, pad=2)

    for ax in axes.ravel():
        ax.axis("off")

    suptitle = f"Synthetic instances - {n_inst} total"
    if title_suffix:
        suptitle += f"  {title_suffix}"
    fig.suptitle(suptitle, fontsize=9, y=1.002)
    fig.tight_layout(rect=[0, 0, 1, 1])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
#  Standalone preview
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    OUT = Path(__file__).resolve().parent / "synthetic_preview"
    OUT.mkdir(parents=True, exist_ok=True)

    insts = generate_all_instances(
        seeds=(42, 123),
        hex_sides=(260,),
        pv_stain_types=("hill", "linear"),
        pp_stain_types=("linear", "linear"),
        noise_levels=("low",),
        modes=("dual",),
        verbose=True,
    )
    out = visualize_instances(insts, OUT / "synthetic_preview.png",
                               title_suffix="synthetic_lobules preview")
    print(f"Saved preview to {out}")
