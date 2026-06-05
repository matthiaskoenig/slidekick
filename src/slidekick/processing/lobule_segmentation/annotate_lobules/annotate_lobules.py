"""
Graph-based lobule annotator - supports RGB brightfield and fluorescence.

Data model
----------
The source of truth is a planar graph:

    vertices         : (N, 2) float array   full-res pixel coordinates
    edges            : (M, 2) int array     index pairs (i, j) into vertices
    edge_is_tissue   : (M,)  bool           edge comes from the auto-detected
                                            tissue mask contour

Lobules are NOT drawn as polygons. The graph is passed through
shapely.ops.polygonize at save/rasterization time; every minimal enclosed
region becomes one face. A face is flagged as "remainder" if more than
TISSUE_EDGE_FRAC_THRESHOLD of its boundary length lies on tissue edges.

On image load the tissue mask is auto-detected using the project's
multi-Otsu detector (from slidekick.processing.lobule_segmentation.lob_utils).
Its contour is subsampled and injected with edge_is_tissue=True.

Image types
-----------
    fluorescence  - multi-channel uint16/float; each channel is colourised
                    using OME channel colours from the file's metadata.
    brightfield   - 1-, 3-, or 4-channel uint8; displayed as RGB; tissue mask
                    built on inverted gray (tissue is dark, background bright).

Auto-detection uses channel count and dtype; override with --image-type.

Interaction
-----------
    left-click        place vertex (snaps to nearest vertex / edge / tissue
                      edge) and draw an edge from the previous chain vertex.
    space / mmb       end current chain.
    x / right-click   delete nearest vertex or edge.
    z                 undo last graph operation.

Keys
----
    d / e / Q     draw | edit-select | quality-mark mode
    H             heal graph (merge coincident, T-split, dedupe)
    s             save
    n / b         next / previous image (autosaves)
    t             toggle snap   T   toggle tissue-boundary visibility
    + / -         snap radius
    v             toggle vertex markers
    f             recompute and show faces (lobules)
    h / ?         help
    q             quit (autosaves)

Saved JSON schema  (written alongside the source file):
    {
        "schema": "graph_v1",
        "level_saved": 2,
        "level_shape": [H, W],
        "full_shape":  [H0, W0],
        "vertices_full_res": [[x, y], ...],
        "edges":              [[i, j], ...],
        "edge_is_tissue":     [true, ...],
        "quality_markers_full_res": [[x, y, q], ...]
    }

Usage
-----
    python output/annotate_lobules.py
    python output/annotate_lobules.py --dir /path/to/slides --level 2
    python output/annotate_lobules.py --file myslide.tif --snap 12
    python output/annotate_lobules.py --tissue-channels 0,4,5
    python output/annotate_lobules.py --image-type brightfield
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
import tifffile

from shapely.geometry import LineString, Polygon as ShPoly, Point
from shapely.ops import polygonize, unary_union

# -- slidekick imports --------------------------------------------------------
# Guard: ensure src/ is on sys.path when the file is run directly as a script.
_SRC = Path(__file__).resolve().parents[4]   # …/annotate_lobules/ -> src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slidekick.io.wsi import read_wsi                                    # CZI conversion + zarr pyramid
from slidekick.io.metadata import Metadata                               # channel colours / names
from slidekick.processing.roi.roi_utils import ensure_grayscale_uint8   # robust grey conversion
from slidekick.processing.lobule_segmentation.lob_utils import (
    detect_tissue_mask_multiotsu,                                        # multi-Otsu tissue mask
)

# -- constants ----------------------------------------------------------------
TISSUE_EDGE_FRAC_THRESHOLD   = 0.80   # face is "remainder" if >80% tissue-edge boundary
TISSUE_SAMPLE_SPACING_LEVEL  = 12.0   # one tissue vertex per ~N level pixels


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def _to_chw(arr: np.ndarray) -> np.ndarray:
    """Normalise any array to (C, H, W) layout."""
    # Strip leading size-1 axes (OME multi-series wrappers, e.g. (1, C, H, W))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return arr[np.newaxis]                        # (H, W)   -> (1, H, W)
    if arr.ndim == 3:
        s = arr.shape
        # (H, W, C): channel-last heuristic - last dim small, both others larger
        if s[-1] <= 16 and s[0] > s[-1] and s[1] > s[-1]:
            return np.moveaxis(arr, -1, 0)            # (H, W, C) -> (C, H, W)
        # Already (C, H, W) or ambiguous: leave as-is
        return arr
    raise ValueError(f"Cannot reshape array of shape {arr.shape} to (C, H, W)")


def _load_level(filepath: Path, level: int):
    """Return ``(data, scale)`` where *data* is ``(C, H, W)`` and *scale* is
    the ratio full-res / level, i.e. ``full_coord = level_coord * scale``.

    CZI files are converted to TIFF via :func:`slidekick.io.wsi.read_wsi`
    before loading.
    """
    # CZI -> TIFF conversion (read_wsi writes the TIFF alongside the CZI)
    if filepath.suffix.lower() == ".czi":
        _, filepath = read_wsi(filepath)

    with tifffile.TiffFile(str(filepath)) as tif:
        series = tif.series[0]
        has_pyramid = len(series.levels) > 1
        full_shape  = series.shape          # e.g. (C, H0, W0) or (H0, W0, C)

        if has_pyramid and level < len(series.levels):
            data  = series.levels[level].asarray()
            scale = float(full_shape[-1]) / float(series.levels[level].shape[-1])
        else:
            data  = series.asarray()
            scale = float(2 ** level)

    data = _to_chw(data)

    # Stride-based downsampling when no real pyramid exists
    if not (has_pyramid and level < len(series.levels)):
        ds = max(1, int(round(scale)))
        if ds > 1:
            data = data[:, ::ds, ::ds].copy()

    return data, float(scale)


def _load_metadata(filepath: Path) -> Metadata:
    """Load a :class:`~slidekick.io.metadata.Metadata` object and enrich it
    from any embedded OME-XML channel information."""
    storage = filepath.with_suffix(".tiff") \
        if filepath.suffix.lower() == ".czi" else filepath
    meta = Metadata(path_original=filepath, path_storage=storage)
    meta.enrich_from_storage(overwrite=False)
    return meta


# ----------------------------------------------------------------------------
# Image-type detection
# ----------------------------------------------------------------------------

def _detect_image_type(data: np.ndarray, metadata: Metadata) -> str:
    """Return ``'brightfield'`` or ``'fluorescence'``.

    Uses :attr:`Metadata.image_type` when set; otherwise falls back to
    a dtype/channel-count heuristic (uint8 with 1–4 channels -> brightfield).
    """
    itype = (metadata.image_type or "").lower()
    if itype in ("brightfield", "rgb", "h&e", "ihc", "he"):
        return "brightfield"
    if itype in ("fluorescence", "if", "multiplex", "mxif"):
        return "fluorescence"

    C = data.shape[0]
    if data.dtype == np.uint8 and C in (1, 3, 4):
        return "brightfield"
    return "fluorescence"


# ----------------------------------------------------------------------------
# Display compositing
# ----------------------------------------------------------------------------

def _robust_uint8(ch: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.5
                  ) -> np.ndarray:
    """Percentile-stretch a single 2-D channel to uint8."""
    ch = ch.astype(np.float32)
    mask = ch > 0
    if not mask.any():
        return np.zeros_like(ch, dtype=np.uint8)
    lo, hi = np.percentile(ch[mask], [lo_p, hi_p])
    if hi <= lo:
        return np.zeros_like(ch, dtype=np.uint8)
    return np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _decode_ome_color(rgba_int: int):
    """Decode a signed OME RGBA int (0xRRGGBBAA) to (r, g, b) uint8 tuple."""
    u = int(rgba_int) if int(rgba_int) >= 0 else int(rgba_int) + 2 ** 32
    return (u >> 24) & 0xFF, (u >> 16) & 0xFF, (u >> 8) & 0xFF


def _composite_rgb(data: np.ndarray, metadata: Metadata,
                   image_type: str = "fluorescence",
                   max_channels: int = 8) -> np.ndarray:
    """Build an ``(H, W, 3)`` uint8 display image from ``(C, H, W)`` data.

    *brightfield* (1 or 3/4 channels, uint8):
        Channels are mapped to greyscale or directly to RGB.

    *fluorescence* (arbitrary channel count):
        Each channel is colourised using the OME channel colour stored in
        *metadata* (populated by :meth:`Metadata.ensure_channel_metadata`).
        Channels beyond *max_channels* are ignored for display speed.
    """
    C, H, W = data.shape

    # -- brightfield ----------------------------------------------------------
    if image_type == "brightfield":
        if C == 1:
            gray = _robust_uint8(data[0])
            return np.stack([gray, gray, gray], axis=-1)
        if C in (3, 4):
            rgb = np.stack([_robust_uint8(data[i]) for i in range(3)], axis=-1)
            return rgb
        # Fallthrough: treat unexpected channel counts as fluorescence

    # -- fluorescence: per-channel colour accumulation ------------------------
    metadata.ensure_channel_metadata(C)
    acc = np.zeros((H, W, 3), dtype=np.float32)

    for ci in range(min(C, max_channels)):
        ch_f = _robust_uint8(data[ci]).astype(np.float32) / 255.0
        rgba_int = metadata.channel_colors.get(ci)
        if rgba_int is not None:
            r, g, b = _decode_ome_color(rgba_int)
        else:
            r, g, b = Metadata._default_palette_rgb(ci)
        acc[..., 0] += ch_f * r
        acc[..., 1] += ch_f * g
        acc[..., 2] += ch_f * b

    mx = acc.max()
    if mx > 0:
        acc *= 255.0 / mx
    return np.clip(acc, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Tissue detection
# ----------------------------------------------------------------------------

def _detect_tissue_contours(data: np.ndarray,
                             image_type: str = "fluorescence",
                             tissue_channels: tuple | None = None):
    """Detect tissue boundary contours from ``(C, H, W)`` data.

    Returns a list of ``(N, 2) float32`` arrays of level-pixel coordinates,
    arc-length-subsampled to roughly one vertex per
    :data:`TISSUE_SAMPLE_SPACING_LEVEL` pixels.

    Uses :func:`slidekick.processing.roi.roi_utils.ensure_grayscale_uint8`
    for grey conversion and
    :func:`slidekick.processing.lobule_segmentation.lob_utils.detect_tissue_mask_multiotsu`
    for thresholding.
    """
    C = data.shape[0]

    # Select channels for tissue detection
    if tissue_channels:
        valid = [i for i in tissue_channels if 0 <= i < C]
        sub = data[valid] if valid else data
    else:
        sub = data  # all channels -> averaged in ensure_grayscale_uint8

    # Robust grey conversion (handles any dtype / layout via slidekick)
    gray = ensure_grayscale_uint8(sub)   # (H, W) uint8

    # Brightfield: tissue is dark -> invert so tissue becomes bright for the
    # multi-Otsu detector (which returns the brightest class as tissue).
    if image_type == "brightfield":
        gray = 255 - gray

    mask = detect_tissue_mask_multiotsu(gray, auto=True)
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = []
    min_area = max(500, int(0.0005 * gray.size))
    for cnt in contours:
        if float(cv2.contourArea(cnt)) < min_area:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue
        # Arc-length subsample
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum   = np.concatenate([[0.0], np.cumsum(dists)])
        total = cum[-1]
        n_samp = max(8, int(np.ceil(total / TISSUE_SAMPLE_SPACING_LEVEL)))
        targets = np.linspace(0.0, total, n_samp, endpoint=False)
        idx = np.clip(np.searchsorted(cum, targets), 0, len(pts) - 1)
        out.append(pts[idx])
    return out


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _project_point_onto_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return ax, ay, 0.0, (px - ax) ** 2 + (py - ay) ** 2
    t  = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return qx, qy, t, (px - qx) ** 2 + (py - qy) ** 2


# ----------------------------------------------------------------------------
# Graph data model
# ----------------------------------------------------------------------------

class AnnotationGraph:
    """Mutable planar graph in FULL-RES coordinates.

    Vertices store ``(x, y)`` in full-resolution pixels. Display code
    divides by *scale* to obtain level coordinates.
    """

    def __init__(self):
        self.vertices:       list[tuple[float, float]] = []
        self.edges:          list[tuple[int, int]]     = []
        self.edge_is_tissue: list[bool]                = []
        # Quality markers: (full_x, full_y, level)  level ∈ {0, 1, 2}
        self.quality_markers: list[tuple[float, float, int]] = []

    # -- mutation ----------------------------------------------------------

    def add_vertex(self, x: float, y: float) -> int:
        self.vertices.append((float(x), float(y)))
        return len(self.vertices) - 1

    def add_edge(self, i: int, j: int, is_tissue: bool = False) -> int | None:
        if i == j:
            return None
        a, b = (i, j) if i < j else (j, i)
        for k, (ea, eb) in enumerate(self.edges):
            if (ea, eb) == (a, b):
                return k
        self.edges.append((a, b))
        self.edge_is_tissue.append(bool(is_tissue))
        return len(self.edges) - 1

    def remove_vertex(self, vi: int):
        if not (0 <= vi < len(self.vertices)):
            return
        self.vertices.pop(vi)
        new_e, new_f = [], []
        for (a, b), fl in zip(self.edges, self.edge_is_tissue):
            if a == vi or b == vi:
                continue
            new_e.append((a - (a > vi), b - (b > vi)))
            new_f.append(fl)
        self.edges, self.edge_is_tissue = new_e, new_f

    def remove_edge(self, ei: int):
        if 0 <= ei < len(self.edges):
            self.edges.pop(ei)
            self.edge_is_tissue.pop(ei)

    def clear_orphan_vertices(self):
        used = {v for e in self.edges for v in e}
        if len(used) == len(self.vertices):
            return
        keep = sorted(used)
        remap = {old: new for new, old in enumerate(keep)}
        self.vertices       = [self.vertices[i] for i in keep]
        self.edges          = [(remap[a], remap[b]) for a, b in self.edges]

    def split_edge_at(self, ei: int, full_x: float, full_y: float) -> int:
        if not (0 <= ei < len(self.edges)):
            return -1
        i, j   = self.edges[ei]
        is_t   = self.edge_is_tissue[ei]
        nv     = self.add_vertex(full_x, full_y)
        a, b   = (i, nv) if i < nv else (nv, i)
        self.edges[ei] = (a, b)
        a2, b2 = (nv, j) if nv < j else (j, nv)
        self.edges.append((a2, b2))
        self.edge_is_tissue.append(is_t)
        return nv

    # -- queries -----------------------------------------------------------

    def nearest_vertex(self, x: float, y: float, scale: float = 1.0):
        if not self.vertices:
            return None, float("inf")
        V = np.asarray(self.vertices, np.float32) / scale
        d = np.linalg.norm(V - np.array([x, y], np.float32), axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def nearest_edge(self, x: float, y: float, scale: float = 1.0):
        if not self.edges:
            return None, 0.0, 0.0, float("inf")
        best = (None, 0.0, 0.0, float("inf"))
        for k, (i, j) in enumerate(self.edges):
            ax, ay = np.array(self.vertices[i]) / scale
            bx, by = np.array(self.vertices[j]) / scale
            qx, qy, _, d2 = _project_point_onto_segment(x, y, ax, ay, bx, by)
            if d2 < best[3]:
                best = (k, qx, qy, d2)
        return best[0], best[1], best[2], float(np.sqrt(best[3]))

    # -- healing -----------------------------------------------------------

    def heal(self, tol_full: float = 12.0, verbose: bool = False) -> dict:
        """Merge coincident vertices, split T-junctions, deduplicate edges."""
        nm = ns = nd = 0
        for _ in range(8):
            a = self._heal_merge_coincident(tol_full)
            b = self._heal_split_edges_at_vertices(tol_full)
            c = self._heal_dedupe_edges()
            nm += a; ns += b; nd += c
            if a == b == c == 0:
                break
        self.clear_orphan_vertices()
        if verbose and (nm or ns or nd):
            print(f"  heal: merged={nm} split={ns} dedup={nd}")
        return dict(merged=nm, split=ns, dedup=nd)

    def _heal_merge_coincident(self, tol: float) -> int:
        n = len(self.vertices)
        if n == 0:
            return 0
        V     = np.asarray(self.vertices, np.float64)
        tol2  = tol * tol
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return False
            if ra < rb: parent[rb] = ra
            else:        parent[ra] = rb
            return True

        did = False
        for i in range(n):
            for j in range(i + 1, n):
                d = V[j] - V[i]
                if d @ d <= tol2:
                    if union(i, j): did = True
        if not did:
            return 0

        roots = [find(i) for i in range(n)]
        unique = sorted(set(roots))
        r2new  = {r: k for k, r in enumerate(unique)}
        remap  = [r2new[roots[i]] for i in range(n)]
        new_vs = []
        for r in unique:
            mem = [i for i in range(n) if roots[i] == r]
            cx  = float(np.mean([V[i, 0] for i in mem]))
            cy  = float(np.mean([V[i, 1] for i in mem]))
            new_vs.append((cx, cy))
        self.vertices = new_vs
        seen, ne, nf = {}, [], []
        for (a, b), fl in zip(self.edges, self.edge_is_tissue):
            na, nb = remap[a], remap[b]
            if na == nb: continue
            key = (min(na, nb), max(na, nb))
            if key in seen:
                if fl: nf[seen[key]] = True
                continue
            seen[key] = len(ne); ne.append(key); nf.append(fl)
        self.edges, self.edge_is_tissue = ne, nf
        return n - len(self.vertices)

    def _heal_split_edges_at_vertices(self, tol: float) -> int:
        tol2 = tol * tol; splits = 0; ei = 0
        while ei < len(self.edges):
            a, b = self.edges[ei]
            ax, ay = self.vertices[a]; bx, by = self.vertices[b]
            dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
            if L2 < 1e-9: ei += 1; continue
            best_v, best_d2 = -1, tol2
            for vi in range(len(self.vertices)):
                if vi in (a, b): continue
                px, py = self.vertices[vi]
                t = ((px - ax) * dx + (py - ay) * dy) / L2
                if not (0.01 < t < 0.99): continue
                qx, qy = ax + t * dx, ay + t * dy
                d2 = (px - qx) ** 2 + (py - qy) ** 2
                if d2 <= best_d2: best_d2 = d2; best_v = vi
            if best_v >= 0:
                is_t = self.edge_is_tissue[ei]
                na, nb = (a, best_v) if a < best_v else (best_v, a)
                self.edges[ei] = (na, nb)
                ma, mb = (best_v, b) if best_v < b else (b, best_v)
                self.edges.append((ma, mb))
                self.edge_is_tissue.append(is_t)
                splits += 1
                continue
            ei += 1
        return splits

    def _heal_dedupe_edges(self) -> int:
        seen, ne, nf, rm = {}, [], [], 0
        for (a, b), fl in zip(self.edges, self.edge_is_tissue):
            key = (min(a, b), max(a, b))
            if key in seen:
                rm += 1
                if fl: nf[seen[key]] = True
                continue
            seen[key] = len(ne); ne.append(key); nf.append(fl)
        if rm: self.edges, self.edge_is_tissue = ne, nf
        return rm

    # -- face extraction ---------------------------------------------------

    def extract_faces(self, scale: float = 1.0) -> list[dict]:
        """Polygonize the graph and classify each face."""
        if len(self.edges) < 3:
            return []
        lines, tissue_segs = [], []
        for (i, j), is_t in zip(self.edges, self.edge_is_tissue):
            ax, ay = np.array(self.vertices[i]) / scale
            bx, by = np.array(self.vertices[j]) / scale
            ls = LineString([(ax, ay), (bx, by)])
            lines.append(ls)
            if is_t: tissue_segs.append(ls)
        try:
            polys = list(polygonize(unary_union(lines)))
        except Exception as exc:
            print(f"  polygonize error: {exc}")
            return []
        tissue_union = unary_union(tissue_segs) if tissue_segs else None
        markers_lv = [(Point(float(mx) / scale, float(my) / scale), int(q))
                      for mx, my, q in self.quality_markers]
        out = []
        for p in polys:
            if not p.is_valid or p.is_empty or p.area < 4:
                continue
            frac_t = 0.0
            if tissue_union is not None:
                try:
                    inter = p.boundary.intersection(tissue_union)
                    if not inter.is_empty:
                        frac_t = float(inter.length / p.boundary.length)
                except Exception:
                    pass
            is_rem = frac_t >= TISSUE_EDGE_FRAC_THRESHOLD
            q_lvl  = None; n_mk = 0
            for mp, lvl in markers_lv:
                if p.contains(mp):
                    n_mk += 1
                    q_lvl = lvl if q_lvl is None else max(q_lvl, lvl)
            out.append(dict(poly_level=p, is_remainder=is_rem,
                            frac_tissue=frac_t, quality=q_lvl or 1,
                            n_markers=n_mk))
        return out


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

def _json_path(filepath: Path) -> Path:
    return filepath.with_suffix(filepath.suffix + ".lobules.json")


def _save_graph(path: Path, g: AnnotationGraph, level_saved: int,
                level_shape, full_shape):
    data = {
        "schema":       "graph_v1",
        "level_saved":  level_saved,
        "level_shape":  [int(level_shape[0]), int(level_shape[1])],
        "full_shape":   [int(full_shape[0]),  int(full_shape[1])],
        "vertices_full_res":       [[float(x), float(y)] for x, y in g.vertices],
        "edges":                   [[int(a), int(b)]     for a, b in g.edges],
        "edge_is_tissue":          [bool(x) for x in g.edge_is_tissue],
        "quality_markers_full_res":[[float(x), float(y), int(q)]
                                    for x, y, q in g.quality_markers],
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def _load_graph(path: Path) -> AnnotationGraph:
    g = AnnotationGraph()
    if not path.exists():
        return g
    with open(path) as fh:
        data = json.load(fh)
    if data.get("schema") == "graph_v1":
        g.vertices       = [(float(p[0]), float(p[1])) for p in data.get("vertices_full_res", [])]
        g.edges          = [(int(e[0]), int(e[1]))     for e in data.get("edges", [])]
        g.edge_is_tissue = [bool(x) for x in data.get("edge_is_tissue", [False] * len(g.edges))]
        g.quality_markers= [(float(p[0]), float(p[1]), int(p[2]) if len(p) >= 3 else 2)
                            for p in data.get("quality_markers_full_res", [])]
        return g
    # Legacy migration: polygons_full_res
    for poly in data.get("polygons_full_res", []):
        s = len(g.vertices)
        for x, y in poly:
            g.add_vertex(float(x), float(y))
        n = len(poly)
        for k in range(n):
            g.add_edge(s + k, s + ((k + 1) % n))
    print(f"  migrated {len(data.get('polygons_full_res', []))} legacy polygons")
    return g


# ----------------------------------------------------------------------------
# Annotator
# ----------------------------------------------------------------------------

class LobuleAnnotator:
    """Interactive matplotlib-based lobule annotator.

    Parameters
    ----------
    files:
        Ordered list of image paths to annotate.
    level:
        Pyramid level to display (0 = full resolution).
    snap_radius_px:
        Snap-to-vertex/edge radius in *level* pixels.
    tissue_channels:
        Indices of channels used for tissue detection. ``None`` uses all.
    image_type:
        ``'brightfield'``, ``'fluorescence'``, or ``'auto'`` (default).
    """

    def __init__(self, files, level: int = 2, snap_radius_px: float = 10.0,
                 tissue_channels: tuple | None = None,
                 image_type: str = "auto"):
        self.files           = list(files)
        self.level           = level
        self.idx             = 0
        self.tissue_channels = tissue_channels
        self._forced_type    = None if image_type == "auto" else image_type

        self.snap_enabled        = True
        self.snap_radius_level   = float(snap_radius_px)
        self.show_vertex_markers = True
        self.show_tissue         = True
        self.mode                = "draw"     # 'draw' | 'select' | 'quality'
        self.selected            = None
        self.quality_place_level = 2

        # Image state
        self.data        = None   # (C, H, W) raw
        self.rgb         = None   # (H, W, 3) display
        self.scale       = 1.0
        self.full_shape  = None
        self.metadata    = None
        self.image_type  = "fluorescence"

        self.graph         = AnnotationGraph()
        self.current_chain: list[int] = []
        self.undo_stack:    list      = []

        # Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.canvas.mpl_connect("button_press_event",  self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",     self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # Mutable artists
        self._edge_lines = self._tissue_lines = None
        self._vertex_scatter = self._snap_indicator = None
        self._rubberband     = self._status_text    = None
        self._quality_scatter  = self._selection_artist = None
        self._face_patches: list = []

        self._load_image(0)
        self._print_help()

    # -- help --------------------------------------------------------------

    def _print_help(self):
        print("\n" + "=" * 60)
        print("Lobule annotator  (RGB / fluorescence)")
        print("=" * 60)
        print("  MODE: [d]raw | [e]dit select | [Q] quality-mark")
        print("  left click    add vertex + edge  (chain)")
        print("  space / mmb   end chain")
        print("  x / rmb       delete nearest vertex or edge")
        print("  z             undo")
        print("  H             heal graph")
        print("  s             save    n/b   next/prev image")
        print("  t/T           toggle snap / tissue boundary")
        print("  +/-           snap radius    v  vertex markers")
        print("  f             show faces     h  help    q  quit")
        print("=" * 60 + "\n")

    # -- IO ----------------------------------------------------------------

    def _load_image(self, idx: int):
        self._end_chain()
        self.idx = idx
        f = self.files[idx]
        print(f"\n[{idx + 1}/{len(self.files)}] {f.name} ...", flush=True)

        data, scale = _load_level(f, self.level)
        self.data   = data
        self.scale  = scale

        # Metadata (channel colours/names)
        self.metadata   = _load_metadata(f)
        self.image_type = (self._forced_type
                           or _detect_image_type(data, self.metadata))
        print(f"  type={self.image_type}  shape={data.shape}  "
              f"dtype={data.dtype}  scale={scale:.1f}x")

        # Display composite
        self.rgb = _composite_rgb(data, self.metadata, self.image_type)
        H, W     = self.rgb.shape[:2]
        self.full_shape = (int(round(H * scale)), int(round(W * scale)))

        # Load or create annotation graph
        self.graph = _load_graph(_json_path(f))
        info = self.graph.heal(tol_full=3.0 * scale)
        if any(info.values()):
            print(f"  healed on load: {info}")

        # Auto-inject tissue contour if not yet present
        if not any(self.graph.edge_is_tissue):
            contours = _detect_tissue_contours(
                self.data, self.image_type, self.tissue_channels)
            n_v = 0
            for cnt in contours:
                s = len(self.graph.vertices)
                for x, y in cnt:
                    self.graph.add_vertex(float(x) * scale, float(y) * scale)
                n = len(cnt)
                for k in range(n):
                    self.graph.add_edge(s + k, s + ((k + 1) % n), is_tissue=True)
                n_v += n
            print(f"  auto-injected {len(contours)} contour(s), {n_v} vertices")

        self.current_chain = []
        self.undo_stack    = []

        # Reset artists and redraw
        self._edge_lines = self._tissue_lines = None
        self._vertex_scatter = self._snap_indicator = None
        self._rubberband = self._status_text = None
        self._face_patches = []

        self.ax.clear()
        self.ax.imshow(self.rgb)
        self.ax.set_title(
            f"[{idx + 1}/{len(self.files)}] {f.name}   "
            f"type={self.image_type}  level={self.level}  "
            f"full={self.full_shape[1]}×{self.full_shape[0]}")
        self._redraw_all()
        self.fig.canvas.draw_idle()

    # -- drawing -----------------------------------------------------------

    def _verts_lv(self) -> np.ndarray:
        if not self.graph.vertices:
            return np.zeros((0, 2), np.float32)
        return np.asarray(self.graph.vertices, np.float32) / self.scale

    def _redraw_edges(self):
        for attr in ("_edge_lines", "_tissue_lines"):
            art = getattr(self, attr, None)
            if art is not None:
                try: art.remove()
                except Exception: pass
            setattr(self, attr, None)
        verts = self._verts_lv()
        if not self.graph.edges or verts.shape[0] == 0:
            return
        norm_segs, tis_segs = [], []
        for (a, b), is_t in zip(self.graph.edges, self.graph.edge_is_tissue):
            seg = [verts[a], verts[b]]
            (tis_segs if is_t else norm_segs).append(seg)
        if norm_segs:
            self._edge_lines = LineCollection(
                norm_segs, colors="lime", linewidths=1.6, zorder=3)
            self.ax.add_collection(self._edge_lines)
        if tis_segs and self.show_tissue:
            self._tissue_lines = LineCollection(
                tis_segs, colors="#00ddff", linewidths=1.8,
                linestyles="dashed", alpha=0.85, zorder=2)
            self.ax.add_collection(self._tissue_lines)

    def _redraw_vertices(self):
        if self._vertex_scatter is not None:
            try: self._vertex_scatter.remove()
            except Exception: pass
            self._vertex_scatter = None
        if not self.show_vertex_markers:
            return
        verts = self._verts_lv()
        if verts.shape[0] == 0:
            return
        is_t = np.zeros(len(verts), bool)
        for (a, b), ft in zip(self.graph.edges, self.graph.edge_is_tissue):
            if ft: is_t[a] = is_t[b] = True
        self._vertex_scatter = self.ax.scatter(
            verts[:, 0], verts[:, 1],
            s=np.where(is_t, 10, 20),
            c=np.where(is_t, "#9ad4ff", "cyan"),
            edgecolor="black", linewidth=0.4, zorder=5)

    def _redraw_faces(self, force: bool = False):
        for p in self._face_patches:
            try: p.remove()
            except Exception: pass
        self._face_patches = []
        if not force:
            return
        info = self.graph.heal(tol_full=3.0 * self.scale)
        faces = self.graph.extract_faces(scale=self.scale)
        q_hi = q_lo = 0
        for f in faces:
            p = f["poly_level"]
            if p.geom_type != "Polygon":
                continue
            q = f["quality"]
            if f["is_remainder"]:
                color, alpha = "#404040", 0.30
            elif q == 2:
                color, alpha = "#ffde4a", 0.40; q_hi += 1
            elif q == 0:
                color, alpha = "#ff6060", 0.30; q_lo += 1
            else:
                color, alpha = "#5bffa0", 0.20
            xs, ys = p.exterior.xy
            patch = MplPolygon(list(zip(xs, ys)), closed=True,
                               facecolor=color, edgecolor="none",
                               alpha=alpha, zorder=1)
            self.ax.add_patch(patch)
            self._face_patches.append(patch)
        n_r = sum(1 for f in faces if not f["is_remainder"])
        n_x = sum(1 for f in faces if     f["is_remainder"])
        print(f"  faces: {n_r} lobules ({q_hi} high, {q_lo} low), {n_x} remainder")

    def _redraw_quality_markers(self):
        if self._quality_scatter is not None:
            try: self._quality_scatter.remove()
            except Exception: pass
            self._quality_scatter = None
        if not self.graph.quality_markers:
            return
        arr = np.asarray([(x, y, q) for x, y, q in self.graph.quality_markers],
                         np.float32)
        qs = arr[:, 2].astype(int)
        self._quality_scatter = self.ax.scatter(
            arr[:, 0] / self.scale, arr[:, 1] / self.scale,
            s=40, marker="*",
            c=np.where(qs == 2, "#ffde4a", np.where(qs == 0, "#ff6060", "#5bffa0")),
            edgecolor="black", linewidth=0.6, zorder=7)

    def _redraw_selection(self):
        if self._selection_artist is not None:
            try: self._selection_artist.remove()
            except Exception: pass
            self._selection_artist = None
        if self.selected is None:
            return
        kind, idx = self.selected
        if kind == "vertex" and 0 <= idx < len(self.graph.vertices):
            vx, vy = self.graph.vertices[idx]
            self._selection_artist, = self.ax.plot(
                [vx / self.scale], [vy / self.scale], "o", markersize=18,
                markerfacecolor="none", markeredgecolor="yellow",
                markeredgewidth=2.5, zorder=8)
        elif kind == "edge" and 0 <= idx < len(self.graph.edges):
            a, b = self.graph.edges[idx]
            ax_, ay_ = self.graph.vertices[a]; bx_, by_ = self.graph.vertices[b]
            self._selection_artist, = self.ax.plot(
                [ax_ / self.scale, bx_ / self.scale],
                [ay_ / self.scale, by_ / self.scale],
                "-", color="yellow", linewidth=3.0, alpha=0.9, zorder=8)

    def _draw_status(self):
        if self._status_text is not None:
            try: self._status_text.remove()
            except Exception: pass
            self._status_text = None
        col = {"draw": "lime", "select": "yellow", "quality": "gold"}
        self._status_text = self.ax.text(
            0.01, 0.99,
            f"MODE={self.mode.upper()}  "
            f"V={len(self.graph.vertices)}  E={len(self.graph.edges)}  "
            f"chain={len(self.current_chain)}  "
            f"snap={'ON' if self.snap_enabled else 'OFF'} r={self.snap_radius_level:.0f}  "
            f"q={self.quality_place_level}",
            transform=self.ax.transAxes,
            color=col.get(self.mode, "white"), fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

    def _redraw_all(self):
        self._redraw_edges(); self._redraw_vertices()
        self._redraw_quality_markers(); self._redraw_selection()
        self._draw_status()

    # -- snapping ----------------------------------------------------------

    def _snap(self, x, y):
        if not self.snap_enabled:
            return float(x), float(y), "free", None
        r = self.snap_radius_level
        vi, dv = self.graph.nearest_vertex(x, y, self.scale)
        if vi is not None and dv <= r:
            vx, vy = self.graph.vertices[vi]
            return vx / self.scale, vy / self.scale, "vertex", vi
        ei, qx, qy, de = self.graph.nearest_edge(x, y, self.scale)
        if ei is not None and de <= r:
            return float(qx), float(qy), "edge", ei
        return float(x), float(y), "free", None

    # -- chain logic -------------------------------------------------------

    def _end_chain(self):
        if self.current_chain:
            self.current_chain = []
            if self._rubberband is not None:
                try: self._rubberband.remove()
                except Exception: pass
                self._rubberband = None

    def _push_undo(self, fn):
        self.undo_stack.append(fn)
        if len(self.undo_stack) > 200:
            self.undo_stack = self.undo_stack[-200:]

    def _click_add_vertex(self, x, y):
        sx, sy, kind, target = self._snap(x, y)
        if kind == "vertex":
            v_idx = target; created = False
        elif kind == "edge":
            v_idx   = self.graph.split_edge_at(target, sx * self.scale, sy * self.scale)
            created = True; print("  T-split edge")
        else:
            v_idx   = self.graph.add_vertex(sx * self.scale, sy * self.scale)
            created = True
        created_edge = False
        if self.current_chain and self.current_chain[-1] != v_idx:
            ek = self.graph.add_edge(self.current_chain[-1], v_idx)
            created_edge = ek is not None
        self.current_chain.append(v_idx)

        def _undo():
            if self.current_chain and self.current_chain[-1] == v_idx:
                self.current_chain.pop()
            if created_edge:
                for k in range(len(self.graph.edges) - 1, -1, -1):
                    a, b = self.graph.edges[k]
                    if (a == v_idx or b == v_idx) and not self.graph.edge_is_tissue[k]:
                        self.graph.remove_edge(k); break
            if created and 0 <= v_idx < len(self.graph.vertices):
                if not any(a == v_idx or b == v_idx for a, b in self.graph.edges):
                    self.graph.remove_vertex(v_idx)
                    self.current_chain = [(c - 1 if c > v_idx else c)
                                         for c in self.current_chain]
        self._push_undo(_undo)

    def _click_delete_nearest(self, x, y):
        vi, dv = self.graph.nearest_vertex(x, y, self.scale)
        ei, _, _, de = self.graph.nearest_edge(x, y, self.scale)
        R = self.snap_radius_level * 1.5
        if vi is not None and dv <= R and (ei is None or dv <= de):
            sv = self.graph.vertices[vi]
            sc = list(self.current_chain)
            self.graph.remove_vertex(vi)
            self.current_chain = [(c - 1 if c > vi else c)
                                  for c in self.current_chain if c != vi]
            self._push_undo(lambda: [self.current_chain.__setitem__(
                slice(None), sc)])  # simplified undo
            print(f"  deleted vertex V={len(self.graph.vertices)}")
        elif ei is not None and de <= R:
            self.graph.remove_edge(ei)
            self.graph.clear_orphan_vertices()
            self._push_undo(lambda: None)
            print(f"  deleted edge E={len(self.graph.edges)}")

    # -- mode helpers ------------------------------------------------------

    def _set_mode(self, m: str):
        self.mode = m; self._end_chain(); self.selected = None
        if self._selection_artist:
            try: self._selection_artist.remove()
            except Exception: pass
            self._selection_artist = None
        print(f"  mode -> {m.upper()}")
        self._draw_status(); self.fig.canvas.draw_idle()

    def _quality_click(self, x, y):
        if self.graph.quality_markers:
            arr = np.asarray([(mx, my) for mx, my, _ in self.graph.quality_markers],
                             np.float32) / self.scale
            d   = np.linalg.norm(arr - np.array([x, y], np.float32), axis=1)
            i   = int(np.argmin(d))
            if d[i] <= self.snap_radius_level:
                old = self.graph.quality_markers.pop(i)
                print(f"  removed quality marker q={old[2]}"); return
        self.graph.quality_markers.append(
            (x * self.scale, y * self.scale, int(self.quality_place_level)))
        print(f"  placed quality marker q={self.quality_place_level}")

    def _select_nearest(self, x, y):
        vi, dv = self.graph.nearest_vertex(x, y, self.scale)
        ei, _, _, de = self.graph.nearest_edge(x, y, self.scale)
        R = self.snap_radius_level * 1.5
        if vi is not None and dv <= R and (not (ei is not None and de <= R) or dv <= de):
            self.selected = ("vertex", vi); print(f"  selected vertex #{vi}")
        elif ei is not None and de <= R:
            self.selected = ("edge", ei); print(f"  selected edge #{ei}")
        else:
            self.selected = None; print("  nothing within reach")

    def _move_selected_vertex(self, x, y):
        if self.selected is None or self.selected[0] != "vertex":
            return
        vi = self.selected[1]
        sx, sy, kind, target = self._snap(x, y)
        if kind == "vertex" and target == vi:
            sx, sy, kind = float(x), float(y), "free"
        if kind == "free":
            self.graph.vertices[vi] = (sx * self.scale, sy * self.scale)
            print(f"  moved vertex #{vi}")
        elif kind == "vertex":
            tv = target
            for k in range(len(self.graph.edges)):
                a, b = self.graph.edges[k]
                if a == vi: a = tv
                if b == vi: b = tv
                self.graph.edges[k] = (min(a, b), max(a, b))
            seen, ne, nf = {}, [], []
            for (a, b), fl in zip(self.graph.edges, self.graph.edge_is_tissue):
                if a == b: continue
                key = (a, b)
                if key in seen: continue
                seen[key] = True; ne.append(key); nf.append(fl)
            self.graph.edges = ne; self.graph.edge_is_tissue = nf
            self.graph.remove_vertex(vi)
            if tv > vi: tv -= 1
            self.selected = ("vertex", tv); print("  merged into existing vertex")
        elif kind == "edge":
            nv = self.graph.split_edge_at(target, sx * self.scale, sy * self.scale)
            for k in range(len(self.graph.edges)):
                a, b = self.graph.edges[k]
                if a == vi: a = nv
                if b == vi: b = nv
                self.graph.edges[k] = (min(a, b), max(a, b))
            seen, ne, nf = {}, [], []
            for (a, b), fl in zip(self.graph.edges, self.graph.edge_is_tissue):
                if a == b: continue
                key = (a, b)
                if key in seen: continue
                seen[key] = True; ne.append(key); nf.append(fl)
            self.graph.edges = ne; self.graph.edge_is_tissue = nf
            self.graph.remove_vertex(vi)
            if nv > vi: nv -= 1
            self.selected = ("vertex", nv); print("  merged into T-junction")

    def _delete_selected(self):
        if self.selected is None: return
        kind, idx = self.selected
        if kind == "vertex":
            self.graph.remove_vertex(idx); print(f"  deleted vertex #{idx}")
        elif kind == "edge":
            self.graph.remove_edge(idx)
            self.graph.clear_orphan_vertices(); print(f"  deleted edge #{idx}")
        self.selected = None

    # -- events ------------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None: return
        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "quality":
            if event.button == 1:
                self._quality_click(x, y)
                self._redraw_quality_markers()
                self._redraw_faces(force=bool(self._face_patches))
                self._draw_status(); self.fig.canvas.draw_idle()
            return

        if self.mode == "select":
            if event.button == 1:
                if self.selected is None or self.selected[0] != "vertex":
                    self._select_nearest(x, y)
                else:
                    self._move_selected_vertex(x, y)
                self._redraw_edges(); self._redraw_vertices()
                self._redraw_selection(); self._draw_status()
                self.fig.canvas.draw_idle()
            elif event.button == 3:
                self.selected = None
                if self._selection_artist:
                    try: self._selection_artist.remove()
                    except Exception: pass
                    self._selection_artist = None
                self._draw_status(); self.fig.canvas.draw_idle()
            return

        # draw mode
        if event.button == 1:
            self._click_add_vertex(x, y)
            self._redraw_edges(); self._redraw_vertices()
            if self._snap_indicator:
                try: self._snap_indicator.remove()
                except Exception: pass
                self._snap_indicator = None
            if self._rubberband:
                try: self._rubberband.remove()
                except Exception: pass
                self._rubberband = None
            self._draw_status(); self.fig.canvas.draw_idle()
        elif event.button == 2:
            self._end_chain(); self._draw_status(); self.fig.canvas.draw_idle()
        elif event.button == 3:
            self._click_delete_nearest(x, y)
            self._redraw_edges(); self._redraw_vertices()
            self._draw_status(); self.fig.canvas.draw_idle()

    def _on_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None: return
        x, y = float(event.xdata), float(event.ydata)
        sx, sy, kind, _ = self._snap(x, y)
        # Update snap indicator
        if self._snap_indicator:
            try: self._snap_indicator.remove()
            except Exception: pass
            self._snap_indicator = None
        if kind != "free":
            col = "magenta" if kind == "vertex" else "orange"
            mk  = "o"       if kind == "vertex" else "s"
            self._snap_indicator, = self.ax.plot(
                [sx], [sy], marker=mk, markersize=14,
                markerfacecolor="none", markeredgecolor=col,
                markeredgewidth=2.0, zorder=6)
        # Update rubber-band
        if self._rubberband:
            try: self._rubberband.remove()
            except Exception: pass
            self._rubberband = None
        if self.current_chain:
            verts = self._verts_lv()
            li = self.current_chain[-1]
            if li < len(verts):
                lx, ly = verts[li]
                self._rubberband, = self.ax.plot(
                    [lx, sx], [ly, sy], "-", color="yellow",
                    linewidth=1.2, alpha=0.7, zorder=4)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        k = event.key or ""; kl = k.lower()

        if k == "d":   self._set_mode("draw"); return
        if k == "e":
            self._set_mode("select" if self.mode != "select" else "draw"); return
        if k == "Q":
            self._set_mode("quality" if self.mode != "quality" else "draw"); return
        if k == "H":
            info = self.graph.heal(tol_full=3.0 * self.scale, verbose=True)
            self._end_chain(); self.selected = None
            if self._selection_artist:
                try: self._selection_artist.remove()
                except Exception: pass
                self._selection_artist = None
            self._redraw_edges(); self._redraw_vertices()
            self._redraw_faces(force=bool(self._face_patches))
            self._draw_status(); self.fig.canvas.draw_idle(); return
        if k == "escape":
            if self.selected:
                self.selected = None
                if self._selection_artist:
                    try: self._selection_artist.remove()
                    except Exception: pass
                    self._selection_artist = None
            else:
                self._end_chain()
            self._draw_status(); self.fig.canvas.draw_idle(); return
        if k == "delete" and self.mode == "select" and self.selected:
            self._delete_selected()
            self._redraw_edges(); self._redraw_vertices()
            if self._selection_artist:
                try: self._selection_artist.remove()
                except Exception: pass
                self._selection_artist = None
            self._draw_status(); self.fig.canvas.draw_idle(); return

        if self.mode == "quality" and k in ("0", "1", "2"):
            self.quality_place_level = int(k)
            self._draw_status(); self.fig.canvas.draw_idle(); return

        if kl == "z":
            if self.undo_stack:
                try: self.undo_stack.pop()()
                except Exception as e: print(f"  undo failed: {e}")
                self._redraw_edges(); self._redraw_vertices()
                self._draw_status(); self.fig.canvas.draw_idle()
        elif kl == "x":
            if self.mode == "select" and self.selected:
                self._delete_selected()
                self._redraw_edges(); self._redraw_vertices()
                if self._selection_artist:
                    try: self._selection_artist.remove()
                    except Exception: pass
                    self._selection_artist = None
            elif event.xdata is not None:
                self._click_delete_nearest(float(event.xdata), float(event.ydata))
                self._redraw_edges(); self._redraw_vertices()
            self._draw_status(); self.fig.canvas.draw_idle()
        elif k == " ":
            self._end_chain(); self._draw_status(); self.fig.canvas.draw_idle()
        elif kl == "s":
            self._save()
        elif kl == "n":
            self._save()
            if self.idx + 1 < len(self.files):
                self._load_image(self.idx + 1)
            else:
                print("  already at last image")
        elif kl == "b":
            self._save()
            if self.idx > 0:
                self._load_image(self.idx - 1)
            else:
                print("  already at first image")
        elif k == "T":
            self.show_tissue = not self.show_tissue
            self._redraw_edges(); self._redraw_vertices()
            self.fig.canvas.draw_idle()
        elif kl == "t":
            self.snap_enabled = not self.snap_enabled
            print(f"  snap {'ON' if self.snap_enabled else 'OFF'}")
            self._draw_status(); self.fig.canvas.draw_idle()
        elif kl in ("+", "="):
            self.snap_radius_level = min(100.0, self.snap_radius_level + 2.0)
            self._draw_status(); self.fig.canvas.draw_idle()
        elif kl in ("-", "_"):
            self.snap_radius_level = max(2.0, self.snap_radius_level - 2.0)
            self._draw_status(); self.fig.canvas.draw_idle()
        elif kl == "v":
            self.show_vertex_markers = not self.show_vertex_markers
            self._redraw_vertices(); self.fig.canvas.draw_idle()
        elif kl == "f":
            self._redraw_faces(force=True); self.fig.canvas.draw_idle()
        elif kl in ("h", "?"):
            self._print_help()
        elif kl == "q":
            self._save(); plt.close(self.fig)

    def _save(self):
        info = self.graph.heal(tol_full=3.0 * self.scale)
        if any(info.values()):
            print(f"  healed on save: {info}")
            if self.current_chain and any(
                    c >= len(self.graph.vertices) for c in self.current_chain):
                self._end_chain()
        p = _json_path(self.files[self.idx])
        _save_graph(p, self.graph,
                    level_saved=self.level,
                    level_shape=self.rgb.shape[:2],
                    full_shape=self.full_shape)
        faces  = self.graph.extract_faces(scale=self.scale)
        n_real = sum(1 for f in faces if not f["is_remainder"])
        n_rem  = sum(1 for f in faces if     f["is_remainder"])
        print(f"  saved V={len(self.graph.vertices)} E={len(self.graph.edges)} "
              f"faces={n_real}(+{n_rem} rem) -> {p.name}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Lobule annotator for brightfield and fluorescence WSI")
    ap.add_argument("--dir",  default=None,
                    help="Directory to search for image files "
                         "(default: <project>/data, recursively)")
    ap.add_argument("--level", type=int, default=2,
                    help="Pyramid level (0=full res, default=2)")
    ap.add_argument("--file",  default=None,
                    help="Restrict to a single filename within --dir")
    ap.add_argument("--snap",  type=float, default=10.0,
                    help="Snap radius in level pixels (default=10)")
    ap.add_argument("--tissue-channels", default=None,
                    help="Comma-separated channel indices for tissue detection, "
                         "e.g. 0,4,5  (default: all channels)")
    ap.add_argument("--image-type", default="auto",
                    choices=["auto", "brightfield", "fluorescence"],
                    help="Override image type detection (default: auto)")
    args = ap.parse_args()

    data_dir = Path(args.dir) if args.dir else Path.cwd()
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    tissue_channels = None
    if args.tissue_channels:
        tissue_channels = tuple(int(x) for x in args.tissue_channels.split(","))

    exts  = {".tif", ".tiff", ".czi"}
    files = sorted(p for p in data_dir.rglob("*")
                   if p.suffix.lower() in exts
                   and not p.name.endswith(".lobules.json"))
    if args.file:
        files = [f for f in files if f.name == args.file]

    if not files:
        print(f"No image files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} image(s) in {data_dir}")
    ann = LobuleAnnotator(files, level=args.level, snap_radius_px=args.snap,
                          tissue_channels=tissue_channels,
                          image_type=args.image_type)
    plt.show()


if __name__ == "__main__":
    main()
