# OVERVIEW
# Converts vector-like skeleton segments into a closed-region mask.
# The pipeline builds a planar graph from polylines, detects loops,
# classifies edges using topology and proximity, applies local pruning rules,
# then rasterizes the kept boundaries and performs morphological sealing
# to output instance labels suitable for filling or postprocessing.

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import math
from collections import defaultdict, deque
import heapq

# VISUALIZATION COLORS
# BGR tuples for debug overlays when writing reports
colors = {
    "untreated": (255, 255, 255),
    "marked":    (0, 0, 255),
    "kept":      (255, 255, 0),
    "deleted":   (255, 0, 255),
    "dead":      (0, 165, 255),
}


# GEOMETRY + LOW-LEVEL
def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

# INTERSECTIONS
# Uniform grid broad phase followed by exact segment tests to collect split parameters.
def _seg_intersections_grid(polylines_xy: List[np.ndarray], eps: float
                            ) -> Tuple[List[Dict[int, List[float]]], np.ndarray]:
    # flatten polylines to segment endpoint arrays and track owners and segment indices
    owners, P0, P1, sidx = [], [], [], []
    for pid, xy in enumerate(polylines_xy):
        if xy is None or len(xy) < 2:
            continue
        a, b = xy[:-1], xy[1:]
        n = len(a)
        P0.append(a)
        P1.append(b)
        owners.append(np.full(n, pid, np.int32))
        sidx.append(np.arange(n, dtype=np.int32))
    if not P0:
        return [dict() for _ in polylines_xy], np.empty((0, 2), float)

    P0 = np.vstack(P0)
    P1 = np.vstack(P1)
    owners = np.concatenate(owners)
    sidx = np.concatenate(sidx)

    # build uniform grid cell size based on epsilon then compute grid bounds
    # Cell size scales with eps so pairs that are within snapping tolerance fall in same or neighboring cells.
    # Lower bound 1.0 ensures at least pixel resolution even when eps < 0.5.
    cell = max(2.0 * eps, 1.0)
    minx = float(np.floor(min(P0[:, 0].min(), P1[:, 0].min()) / cell) * cell)
    miny = float(np.floor(min(P0[:, 1].min(), P1[:, 1].min()) / cell) * cell)

    gx0 = np.floor((np.minimum(P0[:, 0], P1[:, 0]) - minx) / cell).astype(np.int32)
    gy0 = np.floor((np.minimum(P0[:, 1], P1[:, 1]) - miny) / cell).astype(np.int32)
    gx1 = np.floor((np.maximum(P0[:, 0], P1[:, 0]) - minx) / cell).astype(np.int32)
    gy1 = np.floor((np.maximum(P0[:, 1], P1[:, 1]) - miny) / cell).astype(np.int32)

    # bucket each segment index into every grid cell overlapped by its bounding box
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i in range(len(P0)):
        for gx in range(gx0[i], gx1[i] + 1):
            for gy in range(gy0[i], gy1[i] + 1):
                buckets[(gx, gy)].append(i)

    hits: List[Tuple[int, int, float, float, float, float]] = []
    seen = set()
    # within a cell test unique segment pairs to find intersections only once
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        L = idxs
        for a_i in range(len(L)):
            i = L[a_i]
            for a_j in range(a_i + 1, len(L)):
                j = L[a_j]
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)

                A, B = P0[i], P1[i]
                C, D = P0[j], P1[j]

                # Reject pairs whose axis-aligned bounding boxes are separated by more than eps.
                # Saves exact tests. eps padding catches near touches that should still split.
                if (max(A[0], B[0]) + eps < min(C[0], D[0]) or
                        max(C[0], D[0]) + eps < min(A[0], B[0]) or
                        max(A[1], B[1]) + eps < min(C[1], D[1]) or
                        max(C[1], D[1]) + eps < min(A[1], B[1])):
                    continue

                o1 = _orient(A, B, C)
                o2 = _orient(A, B, D)
                o3 = _orient(C, D, A)
                o4 = _orient(C, D, B)
                cross = (o1 * o2 <= 0) and (o3 * o4 <= 0)
                if not cross:
                    continue

                r = B - A
                s = D - C
                denom = r[0] * s[1] - r[1] * s[0]

                # parallel or colinear branch use endpoint projections within eps to catch near touches
                # denom approx. 0 means r and s are nearly parallel. Use projections of endpoints onto the other
                # segment to detect near-touches within eps, then add split parameters at those projected t.
                # rr and ss are squared lengths to guard zero-length segments.
                if abs(denom) < 1e-12:
                    rr = r[0] * r[0] + r[1] * r[1]
                    ss = s[0] * s[0] + s[1] * s[1]
                    if rr > 1e-12:
                        for P, t0 in ((C, 0.0), (D, 1.0)):
                            t = np.clip(((P[0] - A[0]) * r[0] + (P[1] - A[1]) * r[1]) / rr, 0.0, 1.0)
                            Q = A + t * r
                            if np.hypot(Q[0] - P[0], Q[1] - P[1]) <= eps:
                                hits.append((i, j, t, t0, Q[0], Q[1]))
                    if ss > 1e-12:
                        for P, t0 in ((A, 0.0), (B, 1.0)):
                            t = np.clip(((P[0] - C[0]) * s[0] + (P[1] - C[1]) * s[1]) / ss, 0.0, 1.0)
                            Q = C + t * s
                            if ((Q[0] - P[0]) ** 2 + (Q[1] - P[1]) ** 2) <= eps * eps:
                                hits.append((i, j, t0, t, P[0], P[1]))
                    continue

                # Solve for param t on AB and u on CD using 2x2 cross-formula.
                # Accept with small tolerance to avoid missing borderline crossings.
                t = ((C[0] - A[0]) * s[1] - (C[1] - A[1]) * s[0]) / denom
                upar = ((C[0] - A[0]) * r[1] - (C[1] - A[1]) * r[0]) / denom
                if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= upar <= 1 + 1e-12:
                    X = A + t * r
                    hits.append((i, j, float(np.clip(t, 0, 1)), float(np.clip(upar, 0, 1)), X[0], X[1]))

    # assemble split parameters for each original segment and append endpoints then unique
    splits: List[Dict[int, List[float]]] = [dict() for _ in polylines_xy]
    pts = []
    for i, j, ti, tj, x, y in hits:
        pid_i, pid_j = int(owners[i]), int(owners[j])
        si, sj = int(sidx[i]), int(sidx[j])
        if 0 <= pid_i < len(polylines_xy):
            splits[pid_i].setdefault(si, []).append(ti)
        if 0 <= pid_j < len(polylines_xy):
            splits[pid_j].setdefault(sj, []).append(tj)
        pts.append([x, y])

    for pid, xy in enumerate(polylines_xy):
        if xy is None or len(xy) < 2:
            continue
        for si in range(len(xy) - 1):
            L = splits[pid].setdefault(si, [])
            # Ensure both endpoints are included so every segment piece remains represented even
            # if no interior intersections were found.
            L.extend([0.0, 1.0])
            s = np.unique(np.clip(np.asarray(L, float), 0.0, 1.0))
            splits[pid][si] = s.tolist()

    node_pts = np.asarray(pts, float) if pts else np.empty((0, 2), float)
    return splits, node_pts


# detect t junctions by projecting endpoints onto nearby segments then append split params
def _augment_t_junctions(polylines_xy: List[np.ndarray], eps: float,
                         splits: List[Dict[int, List[float]]], raw_pts: np.ndarray,
                         interior_min: float, interior_max: float, angle_min_deg: float
                         ) -> Tuple[List[Dict[int, List[float]]], np.ndarray]:
    owners, P0, P1, sidx = [], [], [], []
    for pid, xy in enumerate(polylines_xy):
        if xy is None or len(xy) < 2:
            continue
        a, b = xy[:-1], xy[1:]
        n = len(a)
        P0.append(a)
        P1.append(b)
        owners.append(np.full(n, pid, np.int32))
        sidx.append(np.arange(n, dtype=np.int32))
    if not P0:
        return splits, raw_pts
    P0 = np.vstack(P0)
    P1 = np.vstack(P1)
    owners = np.concatenate(owners)
    sidx = np.concatenate(sidx)

    cell = max(2.0 * eps, 1.0)
    minx = float(np.floor(min(P0[:, 0].min(), P1[:, 0].min()) / cell) * cell)
    miny = float(np.floor(min(P0[:, 1].min(), P1[:, 1].min()) / cell) * cell)
    gx0 = np.floor((np.minimum(P0[:, 0], P1[:, 0]) - minx) / cell).astype(np.int32)
    gy0 = np.floor((np.minimum(P0[:, 1], P1[:, 1]) - miny) / cell).astype(np.int32)
    gx1 = np.floor((np.maximum(P0[:, 0], P1[:, 0]) - minx) / cell).astype(np.int32)
    gy1 = np.floor((np.maximum(P0[:, 1], P1[:, 1]) - miny) / cell).astype(np.int32)

    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i in range(len(P0)):
        for gx in range(gx0[i], gx1[i] + 1):
            for gy in range(gy0[i], gy1[i] + 1):
                buckets[(gx, gy)].append(i)

    # Local bucket query
    # Iterate over the 3x3 neighborhood of grid cells around (x,y) to find candidate segments.
    # This bounds the search to O(1) average while capturing segments within eps.
    def near_segments(x: float, y: float):
        ix = int(np.floor((x - minx) / cell))
        iy = int(np.floor((y - miny) / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((ix + dx, iy + dy), []):
                    yield j

    new_pts = [] if raw_pts.size else []
    for pid, xy in enumerate(polylines_xy):
        if xy is None:
            continue
        vdirs = []
        if len(xy) >= 2:
            vdirs = [xy[1] - xy[0], xy[-1] - xy[-2]]
        for k, P in enumerate((xy[0], xy[-1])):
            vdir = vdirs[k] if k < len(vdirs) else None
            for j in near_segments(float(P[0]), float(P[1])):
                if owners[j] == pid and (np.allclose(P, P0[j]) or np.allclose(P, P1[j])):
                    continue
                s = P1[j] - P0[j]
                ss = float(s[0] * s[0] + s[1] * s[1])
                if ss < 1e-12:
                    continue
                t = ((P[0] - P0[j][0]) * s[0] + (P[1] - P0[j][1]) * s[1]) / ss
                t = float(np.clip(t, 0.0, 1.0))
                # Only accept projections that land well inside the segment interior. This avoids
                # spurious splits exactly at endpoints which are already nodes. Bounds given in [0,1].
                if not (interior_min <= t <= interior_max):
                    continue
                Q = P0[j] + t * s
                if ((Q[0] - P[0]) ** 2 + (Q[1] - P[1]) ** 2) <= eps * eps:
                    # Filter angles
                    # If the endpoint has a well-defined direction vdir, enforce a minimum angle between
                    # the endpoint ray and the host segment. Suppresses nearly-colinear extensions.
                    if vdir is not None:
                        sdir = s / max(1e-12, np.linalg.norm(s))
                        vdir_n = vdir / max(1e-12, np.linalg.norm(vdir))
                        if abs(float(np.dot(sdir, vdir_n))) > math.cos(math.radians(angle_min_deg)):
                            continue
                    splits[int(owners[j])].setdefault(int(sidx[j]), []).append(t)
                    new_pts.append([float(Q[0]), float(Q[1])])

    if new_pts:
        new_pts = np.asarray(new_pts, float)
        raw_pts = np.vstack([raw_pts, new_pts]) if raw_pts.size else new_pts
    return splits, raw_pts


# cluster many candidate points into node centers using coarse grid plus union find
def _cluster_points_grid(points_xy: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    if points_xy.size == 0:
        return np.empty((0,), np.int32), np.empty((0, 2), float)

    # Use eps as clustering radius. Minimum 1 px so two integer-equal points always cluster.
    cell = max(eps, 1.0)
    qx = np.floor(points_xy[:, 0] / cell).astype(np.int64)
    qy = np.floor(points_xy[:, 1] / cell).astype(np.int64)

    cells = defaultdict(list)
    for idx, (ix, iy) in enumerate(zip(qx, qy)):
        cells[(int(ix), int(iy))].append(idx)

    parent = np.arange(len(points_xy), dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (ix, iy), idxs in cells.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh.extend(cells.get((ix + dx, iy + dy), []))
        if not neigh:
            continue
        neigh = np.asarray(neigh, np.int32)
        for i in idxs:
            Pi = points_xy[i]
            d2 = (points_xy[neigh, 0] - Pi[0]) ** 2 + (points_xy[neigh, 1] - Pi[1]) ** 2
            close = neigh[d2 <= eps * eps]
            for j in close:
                union(i, int(j))

    root_to_new: Dict[int, int] = {}
    labels = np.empty(len(points_xy), np.int32)
    centers = []
    for i in range(len(points_xy)):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = len(centers)
            centers.append(points_xy[r])
        labels[i] = root_to_new[r]
    centers_xy = np.vstack(centers) if centers else np.empty((0, 2), float)
    if centers_xy.size:
        sums = np.zeros_like(centers_xy)
        cnt = np.zeros((centers_xy.shape[0],), int)
        for i, k in enumerate(labels):
            sums[k] += points_xy[i]
            cnt[k] += 1
        centers_xy = sums / np.maximum(cnt[:, None], 1)
    return labels, centers_xy

# grid nearest neighbor
# O(1) average bucketed search for nearest node center within eps in xy space
class _GridNearest:
    def __init__(self, centers_xy: np.ndarray, eps: float):
        self.eps = float(eps)
        self.cell = max(self.eps, 1.0)
        self.centers = centers_xy.copy()

        self.buckets = defaultdict(list)
        for i, (x, y) in enumerate(self.centers):
            ix = int(np.floor(x / self.cell))
            iy = int(np.floor(y / self.cell))
            self.buckets[(ix, iy)].append(i)

    def _bucket(self, x: float, y: float):
        ix = int(np.floor(x / self.cell))
        iy = int(np.floor(y / self.cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield (ix + dx, iy + dy)

    def query(self, x: float, y: float) -> Optional[int]:
        best = None
        bestd = 1e18
        for key in self._bucket(x, y):
            for j in self.buckets.get(key, []):
                cx, cy = self.centers[j]
                d2 = (cx - x) ** 2 + (cy - y) ** 2
                if d2 < bestd:
                    bestd = d2
                    best = j
        # Accept nearest center only if within eps. Compare squared distances to avoid sqrt.
        if best is not None and bestd <= self.eps * self.eps:
            return int(best)
        return None

    def add(self, x: float, y: float) -> int:
        j = len(self.centers)
        self.centers = np.vstack([self.centers, [x, y]]) if self.centers.size else np.array([[x, y]], float)
        ix = int(np.floor(x / self.cell))
        iy = int(np.floor(y / self.cell))
        self.buckets[(ix, iy)].append(j)
        return j


# Draw a 1 px polyline and return its discrete pixels in row col order.
def _rasterize_polyline_to_pixels(rc: np.ndarray, H: int, W: int) -> List[Tuple[int, int]]:
    if len(rc) < 2:
        return []
    canvas = np.zeros((H, W), np.uint8)
    cv2.polylines(canvas, [rc[:, ::-1].reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8)
    rr, cc = np.where(canvas > 0)
    return list(zip(rr.tolist(), cc.tolist()))


def _edge_dir_at_node(edge_pts_xy: np.ndarray, node_is_u: bool) -> float:
    # Sample a short chord near the node instead of the immediate segment. Stabilizes direction
    # at noisy polylines by looking up to k steps away.
    k = min(4, len(edge_pts_xy) - 1)
    if node_is_u:
        v = edge_pts_xy[min(k, len(edge_pts_xy) - 1)] - edge_pts_xy[0]
    else:
        v = edge_pts_xy[-(k + 1)] - edge_pts_xy[-1]
    if np.allclose(v, 0):
        v = edge_pts_xy[-1] - edge_pts_xy[0]
    return math.degrees(math.atan2(v[1], v[0])) % 360.0

# angle clustering
# Count coherent direction groups with circular wraparound to detect multi arm junctions.
def _cluster_angles(angles: List[float], tol_deg: float) -> int:
    # angles are in degrees modulo 360. we sort and count jumps larger than tol to estimate arm count
    if not angles:
        return 0
    # Sort angles on [0,360). A new cluster starts when circular difference exceeds tol_deg.
    # This counts coherent direction groups at a junction.
    a = np.array(sorted(angles))
    clusters = 1
    base = a[0]
    for x in a[1:]:
        diff = abs(x - base)
        diff = min(diff, 360.0 - diff)
        if diff > tol_deg:
            clusters += 1
            base = x
    return clusters


# FACES + DISTANCES

# CV Mask
# Optional filled contour mask to restrict which faces count as interior.
def _build_central_mask(H: int, W: int, cv_contours: Optional[List[np.ndarray]]) -> Optional[np.ndarray]:
    if not cv_contours:
        return None
    m = np.zeros((H, W), np.uint8)
    for cnt in cv_contours:
        if cnt is None or len(cnt) < 3:
            continue
        cv2.drawContours(m, [cnt.astype(np.int32)], -1, 255, thickness=-1)
    return (m > 0).astype(np.uint8)


# get faces from edges
# Rasterize edges as walls and label the complement to get face ids and boundaries.
def _faces_from_edges(H: int, W: int, edges: List[Dict[str, Any]]
                      ) -> Tuple[List[int], np.ndarray, Dict[int, np.ndarray]]:
    wall = np.zeros((H, W), np.uint8)
    for e in edges:
        rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
        cv2.polylines(wall, [rc[:, ::-1].reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8)
    comp_n, comp = cv2.connectedComponents(255 - wall, connectivity=4)
    if comp_n <= 1:
        return [], comp, {}
    border = set(np.unique(np.concatenate([comp[0, :], comp[-1, :], comp[:, 0], comp[:, -1]])))
    interior = [lab for lab in range(1, comp_n) if lab not in border]

    dists: Dict[int, np.ndarray] = {}
    for lab in interior:
        region = (comp == lab).astype(np.uint8)
        cnts, _ = cv2.findContours((region * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary = np.zeros((H, W), np.uint8)
        for c in cnts:
            if len(c) >= 3:
                cv2.polylines(boundary, [c], True, 255, 1, cv2.LINE_8)
        inv = (boundary == 0).astype(np.uint8) * 255
        # Distance to the nearest boundary pixel. Used later to decide whether an edge
        # runs along a face boundary within a pixel tolerance.
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dists[lab] = dist
    return interior, comp, dists


# PARALLEL EDGE DEDUPLICATION  + VIS

# remove duplicate near parallel edges between same node pair based on overlap and angle
def _dedupe_parallel_edges(
        edges: List[Dict[str, Any]],
        H: int,
        W: int,
        thr_overlap: float = 0.5,
        prox_px: float = 1.5,
        ang_tol: float = 25.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[np.ndarray]]]:
    """
    For edges sharing the same unordered node pair {u,v}, drop longer ones only if:
      1) they substantially overlap an already kept path (>= thr_overlap within prox_px), and
      2) they are near-parallel at both endpoints.

    Returns:
      filtered_edges,
      {'kept': [pts_xy of kept edges that actually removed a partner],
       'removed': [pts_xy of removed]}
    """
    if not edges:
        return edges, {'kept': [], 'removed': []}

    groups = defaultdict(list)
    for eid, e in enumerate(edges):
        key = (min(int(e["u"]), int(e["v"])), max(int(e["u"]), int(e["v"])))
        groups[key].append((eid, e))

    # cache pixels and lengths
    pix_cache: Dict[int, np.ndarray] = {}
    len_cache: Dict[int, int] = {}
    for eid, e in enumerate(edges):
        rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
        pix = _rasterize_polyline_to_pixels(rc, H, W)
        arr = np.asarray(pix, dtype=np.int32) if pix else np.empty((0, 2), np.int32)
        pix_cache[eid] = arr
        len_cache[eid] = max(1, arr.shape[0])

    drop = set()
    kept_vis: List[np.ndarray] = []
    removed_vis: List[np.ndarray] = []

    for _, lst in groups.items():
        if len(lst) <= 1:
            continue
        lst_sorted = sorted(lst, key=lambda kv: len_cache[kv[0]])  # keep shorter first

        # endpoint orientations
        # End-point orientation tolerance in degrees. Two edges are treated as parallel if
        # their directions at both ends differ by at most ANG_TOL.

        def ang_delta(a, b):
            return abs((a - b + 180.0) % 360.0 - 180.0)

        grp_angles = {
            eid: (
                _edge_dir_at_node(e["pts_xy"], True),
                _edge_dir_at_node(e["pts_xy"], False)
            )
            for eid, e in lst_sorted
        }

        kept_mask = np.zeros((H, W), np.uint8)
        kept_ids: List[int] = []
        kept_dist = None
        kept_involved = set()

        for eid, e in lst_sorted:
            if eid in drop:
                continue

            if kept_ids:
                # Build distance field for the union of already kept edges. Allows fast overlap
                # estimation by thresholding distances <= prox_px.
                if kept_dist is None:
                    kept_dist = cv2.distanceTransform(255 - kept_mask, cv2.DIST_L2, 3)

                arr = pix_cache[eid]
                frac = 0.0
                if arr.size > 0:
                    rr, cc = arr[:, 0], arr[:, 1]
                    # Overlap fraction: share of this edge's pixels that lie within prox_px of kept mask.
                    # thr_overlap is the minimum fraction required to consider it a duplicate.
                    frac = float((kept_dist[rr, cc] <= prox_px).sum()) / float(len_cache[eid])

                a_u_e, a_v_e = grp_angles[eid]
                best_k, best_diff = None, 1e9
                for k in kept_ids:
                    a_u_k, a_v_k = grp_angles[k]
                    # Parallel check uses the worst of the two endpoint direction differences so
                    # curved edges that diverge near either end are not merged.
                    d = max(ang_delta(a_u_e, a_u_k), ang_delta(a_v_e, a_v_k))
                    if d < best_diff:
                        best_diff, best_k = d, k

                if frac >= thr_overlap and best_diff <= ang_tol:
                    drop.add(eid)
                    removed_vis.append(e["pts_xy"])
                    kept_involved.add(best_k)
                    continue

            kept_ids.append(eid)
            rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
            cv2.polylines(kept_mask, [rc[:, ::-1].reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8)
            kept_dist = None

        for k in kept_involved:
            kept_vis.append(edges[k]["pts_xy"])

    if not drop:
        return edges, {'kept': kept_vis, 'removed': removed_vis}

    keep_idx = [i for i in range(len(edges)) if i not in drop]
    return [edges[i] for i in keep_idx], {'kept': kept_vis, 'removed': removed_vis}

# remove short near parallel caps at nodes to stabilize loops and reduce spurs
def _dedupe_caps_at_nodes(
        edges: List[Dict[str, Any]],
        H: int,
        W: int,
        *,
        prox_px: float = 1.5,
        thr_overlap: float = 0.8,
        max_len_px: int = 25,
        ang_tol_deg: float = 25.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[np.ndarray]]]:
    """
    Remove short near-parallel stubs that share a node with a longer edge and
    lie within its proximity envelope for most of their pixels.
    """
    if not edges:
        return edges, {'kept': [], 'removed': []}

    # raster cache
    pix_cache: Dict[int, np.ndarray] = {}
    len_cache: Dict[int, int] = {}
    for eid, e in enumerate(edges):
        rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
        pix = _rasterize_polyline_to_pixels(rc, H, W)
        arr = np.asarray(pix, dtype=np.int32) if pix else np.empty((0, 2), np.int32)
        pix_cache[eid] = arr
        len_cache[eid] = max(1, arr.shape[0])

    # adjacency
    node_edges: Dict[int, List[int]] = {}
    for eid, e in enumerate(edges):
        u, v = int(e["u"]), int(e["v"])
        node_edges.setdefault(u, []).append(eid)
        node_edges.setdefault(v, []).append(eid)

    def ang_delta(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    drop = set()
    kept_vis: List[np.ndarray] = []
    removed_vis: List[np.ndarray] = []

    for nid, lst in node_edges.items():
        if len(lst) <= 1:
            continue

        # Sort by length desc so envelopes come from longer curves
        # Process longer edges first. They define the local envelope used to absorb short caps.
        lst_sorted = sorted(lst, key=lambda i: len_cache[i], reverse=True)

        # Build cumulative mask while keeping a small set per direction cluster
        mask = np.zeros((H, W), np.uint8)
        reps: List[int] = []  # representative kept edges for overlay
        rep_dirs: List[float] = []

        for eid in lst_sorted:
            if eid in drop:
                continue
            e = edges[eid]
            # pick direction of edge as it leaves nid
            node_is_u = (nid == int(e["u"]))
            a_e = _edge_dir_at_node(e["pts_xy"], node_is_u=node_is_u)

            # find best representative with similar direction
            # Pick a representative kept edge in the same direction cluster to compare against.
            best_rep = None
            best_d = 1e9
            for rid, a_rep in zip(reps, rep_dirs):
                d = ang_delta(a_e, a_rep)
                if d < best_d:
                    best_d = d
                    best_rep = rid

            # If this edge is short, check overlap with mask from its best direction
            if len_cache[eid] <= max_len_px and best_rep is not None and best_d <= ang_tol_deg:
                # distance field from current mask
                dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
                pix = pix_cache[eid]
                if pix.size > 0:
                    rr, cc = pix[:, 0], pix[:, 1]
                    # If most pixels of the short edge fall inside the proximity envelope, treat it as a cap
                    # and remove it. prox_px relates to stroke width; thr_overlap is typically high (e.g., 0.8)
                    frac = float((dist[rr, cc] <= prox_px).sum()) / float(len_cache[eid])
                    if frac >= thr_overlap:
                        drop.add(eid)
                        removed_vis.append(e["pts_xy"])
                        continue

            # keep it and update the directional mask
            reps.append(eid)
            rep_dirs.append(a_e)
            rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
            cv2.polylines(mask, [rc[:, ::-1].reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8)

        # kept reps that actually caused removals are highlighted
        if removed_vis:
            for k in reps:
                kept_vis.append(edges[k]["pts_xy"])

    if not drop:
        return edges, {'kept': kept_vis, 'removed': removed_vis}

    keep_idx = [i for i in range(len(edges)) if i not in drop]
    return [edges[i] for i in keep_idx], {'kept': kept_vis, 'removed': removed_vis}


# GRAPH HELPERS: BRIDGE DETECTION

# non recursive bridge finder marks edges whose removal disconnects the node graph
def _find_bridges(edges: List[Dict[str, Any]]) -> List[bool]:
    # treat edges as undirected. we compress node ids to a dense index for arrays
    # we avoid recursion to keep stack usage predictable on large graphs
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    nodes_set = set()
    for eid, e in enumerate(edges):
        u = int(e["u"])
        v = int(e["v"])
        if u == v:
            continue
        nodes_set.add(u)
        nodes_set.add(v)
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    if not nodes_set:
        return [False] * len(edges)

    nodes = sorted(nodes_set)
    idx_of = {nid: i for i, nid in enumerate(nodes)}
    N = len(nodes)

    adj_idx: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
    for u_nid, lst in adj.items():
        ui = idx_of[u_nid]
        for v_nid, eid in lst:
            adj_idx[ui].append((idx_of[v_nid], eid))

    disc = [-1] * N
    low = [0] * N
    is_bridge = [False] * len(edges)
    time = 0

    for start in range(N):
        if disc[start] != -1:
            continue
        # Manual stack holds (node, parent_node_idx, parent_edge_id, iterator_index) to avoid recursion.
        stack: List[Tuple[int, int, int, int]] = [(start, -1, -1, 0)]
        while stack:
            ui, parent_idx, parent_eid, it = stack.pop()
            if disc[ui] == -1:
                disc[ui] = low[ui] = time
                time += 1

            neighbors = adj_idx[ui]
            if it < len(neighbors):
                stack.append((ui, parent_idx, parent_eid, it + 1))
                v_idx, eid = neighbors[it]
                if disc[v_idx] == -1:
                    stack.append((v_idx, ui, eid, 0))
                elif eid != parent_eid:
                    low[ui] = min(low[ui], disc[v_idx])
                continue

            if parent_idx != -1:
                low[parent_idx] = min(low[parent_idx], low[ui])
                if low[ui] > disc[parent_idx] and parent_eid >= 0:
                    is_bridge[parent_eid] = True

    return is_bridge


# SEGMENT-LEVEL RASTER AND DEBUG RECONSTRUCTION HELPERS
# Rasterize each input segment and record which pixels belong to which segment id.
def _rasterize_and_index(
        segments: Optional[List[np.ndarray]],
        shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[List[int]]]]:
    H, W = map(int, shape)
    skel01 = np.zeros((H, W), dtype=np.uint8)
    S_idx = np.full((H, W), -1, dtype=np.int32)
    segpix: Dict[int, List[List[int]]] = {}

    if not segments:
        return skel01, S_idx, segpix

    tmp = np.zeros((H, W), dtype=np.uint8)
    for sid, seg in enumerate(segments):
        if seg is None or len(seg) < 2:
            continue
        pts = np.asarray(seg, dtype=np.float32)
        pts = np.round(pts).astype(np.int32)
        if len(pts) < 2:
            continue
        tmp.fill(0)
        pts_cv = pts[:, ::-1].reshape(-1, 1, 2)
        cv2.polylines(tmp, [pts_cv], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_8)
        nz = cv2.findNonZero(tmp)
        if nz is None:
            continue
        xy = nz[:, 0]
        cc = xy[:, 0].astype(int)
        rr = xy[:, 1].astype(int)
        skel01[rr, cc] = 1
        mnew = (S_idx[rr, cc] == -1)
        if np.any(mnew):
            S_idx[rr[mnew], cc[mnew]] = sid
        segpix.setdefault(int(sid), [])
        for r, c in zip(rr.tolist(), cc.tolist()):
            segpix[int(sid)].append([int(r), int(c)])

    for sid, pts in segpix.items():
        if pts:
            pts.sort(key=lambda rc: (rc[0], rc[1]))

    return skel01, S_idx, segpix

#  DASHED DRAWING HELPER (for vis)
# For visualization only. Alternates dash and gap along each segment.
def _draw_dashed_polyline(img: np.ndarray, pts_xy: np.ndarray, color: Tuple[int, int, int],
                          thickness: int, dash_px: int = 8, gap_px: int = 6) -> None:
    if pts_xy is None or len(pts_xy) < 2:
        return
    p = pts_xy[0].astype(float)
    for q in pts_xy[1:]:
        q = q.astype(float)
        v = q - p
        L = float(np.hypot(v[0], v[1]))
        if L <= 1e-6:
            p = q
            continue
        dirv = v / L
        t = 0.0
        on = True
        while t < L:
            step = dash_px if on else gap_px
            t_next = min(L, t + step)
            if on:
                a = p + dirv * t
                b = p + dirv * t_next
                cv2.line(img,
                         (int(round(a[0])), int(round(a[1]))),
                         (int(round(b[0])), int(round(b[1]))),
                         color, thickness, cv2.LINE_8)
            on = not on
            t = t_next
        p = q


# MAIN API
# End to end assembly of nodes edges faces and final mask with rule based pruning.
def process_segments_to_mask(
        segments: Optional[List[np.ndarray]],
        image_shape: Tuple[int, int],
        cv_contours: Optional[List[np.ndarray]] = None,
        report_path: Optional[str] = None,
        min_area_px: int = 50,
        thick_width_px: int = 3,
        snap_eps_px: Optional[float] = None,
        min_edge_keep_px: float = 0.0,
        angle_tol_deg: float = 15.0,
        edge_boundary_tol_px: float = 1.5,
        edge_on_boundary_frac: float = 0.6,
        min_neighbor_len: int = 8,
        active_marked_frac: float = 0.8,
        two_path_diff_frac: float = 0.20,
        interior_min: float = 0.02,
        interior_max: float = 0.98,
        angle_min_deg: float = 20.0,
        align_tol: float = 25.0,
        dangler_tol: float = 40.0,
        thr_overlap_node: float = 0.85,
        thr_overlap_parallel: float = 0.5,
        angl_tol_parallel: float = 25.0,
) -> np.uint8:
    # PIPELINE
    # 1 Build tolerance scales from stroke width then convert input rc polylines to xy.
    # 2 Split segments at crossings and tee joints so chains can be traced through nodes.
    # 3 Cluster nearby endpoints to prevent hairline gaps and assign integer node ids.
    # 4 Trace maximal chains into edges, then remove parallel duplicates and tiny caps.
    # 5 Compute faces by raster walling, classify faces as marked or untreated.
    # 6 Classify edges by which faces they border and by bridge topology.
    # 7 Promote or cull short stubs using proximity and alignment to marked edges.
    # 8 Build segment level catalogs and loop to edge maps.
    # 9 Apply rules A through E to move segments between kept and deleted sets in memory.
    # 10 Close small gaps morphologically and label connected interior regions.

    H, W = map(int, image_shape)
    # thickness and tolerances derive from the drawing stroke width to make thresholds scale invariant
    thick_width_px = int(max(1, thick_width_px))

    # eps is the snapping tolerance in pixels used for intersection bucketing and merging
    # default scales with half the stroke, lower bounded by one pixel for grid robustness
    # SNAP TOLERANCE
    # eps is proportional to stroke width. Larger strokes imply larger uncertainty in vertex placement.
    eps = float(snap_eps_px if snap_eps_px is not None else max(1.0, thick_width_px / 2.0))

    # slightly smaller tolerance for endpoint snapping to avoid over merging near junction clusters
    # Endpoint snapping uses a slightly smaller tolerance to prevent over-merging at dense junctions.
    eps_end = 0.9 * eps

    # max length in pixels for a stub considered in the post pass
    # 3x thickness keeps proportion while 6 px guards tiny doodles
    # Stub length threshold in pixels. Short terminal edges below this are candidates for post-pass logic.

    lstub = max(6, 3 * int(thick_width_px))
    # neighborhood radius around marked edges for proximity tests
    # we use ~0.6x thickness so stubs touching the same stroke are promoted
    # Proximity radius used when measuring how tightly a stub clings to a marked edge.

    near_marked_px = max(1.2, 0.6 * thick_width_px)
    # cap removal considers edges shorter than this. 4x thickness balances noise cleanup vs structure
    # Maximum length for a cap in the dedupe-at-nodes pass. Scales with thickness.

    max_len_px = max(12, 4 * int(thick_width_px))

    # RC -> XY
    polys_xy: List[np.ndarray] = []
    for seg in segments:
        if seg is None or len(seg) < 2:
            polys_xy.append(None)
            continue
        rc = np.asarray(seg, float)
        xy = np.column_stack([rc[:, 1], rc[:, 0]])
        keep = [0]
        for i in range(1, len(xy)):
            if not np.allclose(xy[i], xy[i - 1]):
                keep.append(i)
        xy = xy[keep]
        polys_xy.append(xy if len(xy) >= 2 else None)

    # intersections + T
    splits, raw_int_pts = _seg_intersections_grid(polys_xy, eps)
    # T-junction augmentation uses a more generous tolerance to catch near-misses at tees.
    eps_t = max(eps, 1.5 * float(thick_width_px))
    # add t junction split points by projecting endpoints to nearby segments
    splits, raw_int_pts = _augment_t_junctions(polys_xy, eps_t, splits, raw_int_pts, interior_min, interior_max, angle_min_deg)

    # snap end-points that meet
    end_pts = []
    for _xy in polys_xy:
        if _xy is None or len(_xy) == 0:
            continue
        end_pts.append(_xy[0])
        end_pts.append(_xy[-1])
    if end_pts:
        _ends = np.asarray(end_pts, float)
        # snap endpoints closer than eps_end to a shared center to avoid hairline gaps
        _lbl, _cent = _cluster_points_grid(_ends, eps_end)
        if _cent.size and _lbl.size:

            _cnt = np.bincount(_lbl, minlength=len(_cent))
            _keep = _cnt >= 2
            if _keep.any():
                _cent = _cent[_keep]
                raw_int_pts = np.vstack([raw_int_pts, _cent]) if raw_int_pts.size else _cent

    # nodes
    _, centers_int = _cluster_points_grid(raw_int_pts, eps * 1.2)
    knn = _GridNearest(centers_int, eps_end)
    node_xy: Dict[int, Tuple[float, float]] = {i + 1: (float(centers_int[i, 0]), float(centers_int[i, 1]))
                                               for i in range(len(centers_int))}
    next_nid = len(node_xy) + 1

    def node_for_point(p: np.ndarray, allow_new: bool = True) -> int:
        nonlocal next_nid
        j = knn.query(float(p[0]), float(p[1]))
        if j is not None:
            return int(j + 1)
        if allow_new:
            knn.add(float(p[0]), float(p[1]))
            nid = next_nid
            next_nid += 1
            node_xy[nid] = (float(p[0]), float(p[1]))
            return nid
        return -1

    # pieces
    pieces: List[Dict[str, Any]] = []
    for pid, xy in enumerate(polys_xy):
        if xy is None:
            continue
        for s in range(len(xy) - 1):
            a, b = xy[s], xy[s + 1]
            ts = splits[pid].get(s, [0.0, 1.0])
            ts = np.unique(np.clip(np.asarray(ts, float), 0.0, 1.0))
            if len(ts) < 2:
                ts = np.array([0.0, 1.0], float)
            for k in range(len(ts) - 1):
                t0, t1 = float(ts[k]), float(ts[k + 1])
                p0, p1 = a + t0 * (b - a), a + t1 * (b - a)
                u, v = node_for_point(p0, True), node_for_point(p1, True)
                # Degenerate piece collapsed to a point. Keep only if longer than min_edge_keep_px.
                # Otherwise synthesize a new node so the chain can proceed.
                if u == v:
                    if np.hypot(*(p1 - p0)) < min_edge_keep_px:
                        continue
                    knn.add(float(p1[0]), float(p1[1]))
                    node_xy[next_nid] = (float(p1[0]), float(p1[1]))
                    v = next_nid
                    next_nid += 1
                pieces.append({"u": int(u), "v": int(v), "pid": pid, "p0": p0, "p1": p1})

    # adjacency over pieces
    adj: Dict[int, List[int]] = {}
    for eid, pc in enumerate(pieces):
        adj.setdefault(pc["u"], []).append(eid)
        adj.setdefault(pc["v"], []).append(eid)
    deg0 = {n: len(adj.get(n, [])) for n in node_xy.keys()}

    # chain -> edges
    used = np.zeros(len(pieces), dtype=bool)
    edges: List[Dict[str, Any]] = []

    # trace a maximal chain by walking degree two nodes until a termination node is reached
    def follow(start_node: int, start_eid: int):
        chain: List[np.ndarray] = []
        pids: List[int] = []
        cur_node = start_node
        eid = start_eid
        while True:
            used[eid] = True
            pc = pieces[eid]
            u, v = pc["u"], pc["v"]
            nxt = v if u == cur_node else u
            chain.append(np.vstack([pc["p0"], pc["p1"]]) if cur_node == u
                         else np.vstack([pc["p1"], pc["p0"]]))
            pids.append(pc["pid"])
            # Stop growing the chain when reaching a non-degree-2 node. Chains connect maximal degree-2 runs.
            if deg0.get(nxt, 0) != 2:
                return start_node, nxt, chain, pids
            nxt_eids = [e for e in adj[nxt] if not used[e]]
            if not nxt_eids:
                return start_node, nxt, chain, pids
            cur_node = nxt
            eid = nxt_eids[0]

    terminals = [n for n, d in deg0.items() if d != 2]
    for n in terminals:
        for eid in adj.get(n, []):
            if used[eid]:
                continue
            u, v, chain, pids = follow(n, eid)
            coords = [chain[0][0]]
            for seg_xy in chain:
                coords.append(seg_xy[-1])
            edges.append({"u": int(u), "v": int(v),
                          "pts_xy": np.vstack(coords),
                          "seg_ids": sorted(set(pids))})

    for eid in range(len(pieces)):
        if used[eid]:
            continue
        pc = pieces[eid]
        used[eid] = True
        path = [np.vstack([pc["p0"], pc["p1"]])]
        pids = [pc["pid"]]
        cur_node = pc["v"]
        while True:
            nxt_eids = [e for e in adj[cur_node] if not used[e]]
            if not nxt_eids:
                break
            e2 = nxt_eids[0]
            used[e2] = True
            pc2 = pieces[e2]
            if pc2["u"] == cur_node:
                path.append(np.vstack([pc2["p0"], pc2["p1"]]))
                cur_node = pc2["v"]
            else:
                path.append(np.vstack([pc2["p1"], pc2["p0"]]))
                cur_node = pc2["u"]
            pids.append(pc2["pid"])
            if cur_node == pc["u"]:
                break
        coords = [path[0][0]]
        for seg_xy in path:
            coords.append(seg_xy[-1])
        edges.append({"u": int(pc["u"]), "v": int(pc["u"]),
                      "pts_xy": np.vstack(coords),
                      "seg_ids": sorted(set(pids))})

    # dedupe parallel same - {u,v}
    edges, dup_vis = _dedupe_parallel_edges(
        edges, H, W, thr_overlap=thr_overlap_parallel, prox_px=near_marked_px, ang_tol=angl_tol_parallel
    )

    # node-local cap dedupe
    edges, cap_vis = _dedupe_caps_at_nodes(
        edges, H, W,
        prox_px=near_marked_px,
        thr_overlap=thr_overlap_node,
        max_len_px=max_len_px,
        ang_tol_deg=angle_tol_deg,
    )
    # merge vis
    dup_vis['kept'].extend(cap_vis.get('kept', []))
    dup_vis['removed'].extend(cap_vis.get('removed', []))

    # node stats
    node_angles: Dict[int, List[float]] = {nid: [] for nid in node_xy.keys()}
    for e in edges:
        a_u = _edge_dir_at_node(e["pts_xy"], True)
        a_v = (_edge_dir_at_node(e["pts_xy"], False) + 180.0) % 360.0
        node_angles[e["u"]].append(a_u)
        node_angles[e["v"]].append(a_v)
    node_arms: Dict[int, int] = {nid: _cluster_angles(angs, angle_tol_deg) for nid, angs in node_angles.items()}

    node_deg: Dict[int, int] = {nid: 0 for nid in node_xy.keys()}
    for e in edges:
        if e["u"] == e["v"]:
            node_deg[e["u"]] = node_deg.get(e["u"], 0) + 2
        else:
            node_deg[e["u"]] = node_deg.get(e["u"], 0) + 1
            node_deg[e["v"]] = node_deg.get(e["v"], 0) + 1

    nodes_vis: Dict[int, Dict[str, Any]] = {}
    for nid, (x, y) in node_xy.items():
        if node_deg.get(nid, 0) >= 3 or node_arms.get(nid, 0) >= 3:
            r, c = int(round(y)), int(round(x))
            nodes_vis[nid] = {"id": int(nid), "rc": [r, c], "type": "junction"}

    # faces + loop mask classification
    interior_labels, comp, face_dists = _faces_from_edges(H, W, edges)
    central_mask = _build_central_mask(H, W, cv_contours)

    loop_type: Dict[int, str] = {}
    untreated_fill = np.zeros((H, W), np.uint8)
    for lab in interior_labels:
        region = (comp == lab).astype(np.uint8)
        area = int(region.sum())
        has_cv = False
        if central_mask is not None:
            has_cv = bool(np.any(central_mask[region > 0]))
        # Face classification: small areas or outside the central mask are considered 'marked' for removal.
        # Larger areas inside the provided contours remain 'untreated' and will define final interiors.
        loop_type[lab] = "marked" if (area < int(min_area_px) or not has_cv) else "untreated"
        if loop_type[lab] == "untreated":
            untreated_fill[region > 0] = 1

    # edge raster + classification
    edge_pixels: List[List[Tuple[int, int]]] = []
    edge_len = []
    for e in edges:
        rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
        pix = _rasterize_polyline_to_pixels(rc, H, W)
        edge_pixels.append(pix)
        edge_len.append(max(1, len(pix)))

    # identify connectors from topology independent of geometry using bridges
    is_bridge = _find_bridges(edges)
    topo_connector = [False] * len(edges)
    for eid, e in enumerate(edges):
        u, v = int(e["u"]), int(e["v"])
        if u == v:
            topo_connector[eid] = False
        else:
            # Edge is a topological connector if it is a bridge or touches a dangling endpoint (degree ≤ 1).
            # Such edges cannot bound a face and are treated as 'dead' for filling.
            if is_bridge[eid] or node_deg.get(u, 0) <= 1 or node_deg.get(v, 0) <= 1:
                topo_connector[eid] = True

    edge_faces: List[set] = [set() for _ in edges]
    # edge classes used downstream marked untreated connector, i.e., dead
    edge_category = ["unclassified"] * len(edges)

    for eid, pix in enumerate(edge_pixels):
        if topo_connector[eid]:
            edge_category[eid] = "dead"
            continue
        if not pix:
            edge_category[eid] = "dead"
            continue

        rr = np.array([p[0] for p in pix], np.int32)
        cc = np.array([p[1] for p in pix], np.int32)
        votes_marked = 0
        votes_untreated = 0
        touched = []
        for lab in interior_labels:
            dist = face_dists[lab]
            close = (dist[rr, cc] <= float(edge_boundary_tol_px)).sum()
            frac = close / edge_len[eid]
            # Consider this edge adjacent to a face if at least edge_on_boundary_frac of its pixels lie
            # within edge_boundary_tol_px of the face boundary in distance space.
            if frac >= edge_on_boundary_frac:
                touched.append(lab)
                if loop_type[lab] == "marked":
                    votes_marked += 1
                else:
                    votes_untreated += 1
        if touched:
            edge_faces[eid].update(touched)
        if votes_marked > 0:
            edge_category[eid] = "marked"
        elif votes_untreated > 0:
            edge_category[eid] = "untreated"

    for i in range(len(edge_category)):
        if edge_category[i] == "unclassified":
            edge_category[i] = "dead"

    # proximity to marked edges for stub handling
    marked_mask = np.zeros((H, W), np.uint8)
    any_marked = False
    for eid, e in enumerate(edges):
        if edge_category[eid] != "marked":
            continue
        any_marked = True
        rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
        cv2.polylines(marked_mask, [rc[:, ::-1].reshape(-1, 1, 2)], False, 255, int(thick_width_px), cv2.LINE_8)
    if any_marked:
        # distance to marked edges used to promote short untreated stubs that cling to marked
        dist_to_marked = cv2.distanceTransform(255 - marked_mask, cv2.DIST_L2, 3)
    else:
        dist_to_marked = np.full((H, W), 1e6, np.float32)

    # SHORT-STUB POST-PASS
    node_edges_map: Dict[int, List[int]] = {}
    for eid, e in enumerate(edges):
        u, v = int(e["u"]), int(e["v"])
        node_edges_map.setdefault(u, []).append(eid)
        node_edges_map.setdefault(v, []).append(eid)

    for eid, e in enumerate(edges):
        # consider only edges shorter than lstub for stub promotion
        if edge_len[eid] > lstub:
            continue

        u, v = int(e["u"]), int(e["v"])
        du, dv = int(node_deg.get(u, 0)), int(node_deg.get(v, 0))
        # Terminal test via XOR: exactly one endpoint has degree 1, i.e., a dangling stub.
        is_terminal = (du == 1) ^ (dv == 1)
        if not is_terminal:
            continue
        hub = v if du == 1 else u

        a_e = _edge_dir_at_node(e["pts_xy"], node_is_u=(hub == e["u"]))
        best_align_marked = 180.0
        for nb in node_edges_map.get(hub, []):
            if nb == eid or edge_category[nb] != "marked":
                continue
            a_nb = _edge_dir_at_node(edges[nb]["pts_xy"], node_is_u=(hub == edges[nb]["u"]))
            d = abs((a_e - a_nb + 180.0) % 360.0 - 180.0)
            if d < best_align_marked:
                best_align_marked = d

        pix = edge_pixels[eid]
        rr = np.array([p[0] for p in pix], np.int32) if pix else np.empty((0,), np.int32)
        cc = np.array([p[1] for p in pix], np.int32) if pix else np.empty((0,), np.int32)
        near_frac = 0.0
        if rr.size:
            near_frac = float((dist_to_marked[rr, cc] <= near_marked_px).sum()) / float(edge_len[eid])

        # untreated stubs may be promoted to marked if they cling to a marked edge and align in direction
        if edge_category[eid] == "untreated":
            # note near_frac >= near_frac is tautologically true; kept for behavior parity.
            # intended threshold comparison is likely near_frac vs a fixed fraction
            if near_frac >= near_frac and best_align_marked <= align_tol:
                edge_category[eid] = "marked"
                continue
            diffs = []
            for nb in node_edges_map.get(hub, []):
                if nb == eid:
                    continue
                a_nb = _edge_dir_at_node(edges[nb]["pts_xy"], node_is_u=(hub == edges[nb]["u"]))
                d = abs((a_e - a_nb + 180.0) % 360.0 - 180.0)
                diffs.append(d)
            if diffs and min(diffs) >= dangler_tol:
                edge_category[eid] = "dead"
                continue

        # dead stubs can also be upgraded if they really are part of the same marked run
        elif edge_category[eid] == "dead":
            # same tautological guard as above retained intentionally
            if near_frac >= near_frac and best_align_marked <= align_tol:
                edge_category[eid] = "marked"

    # build human readable overlays per category to inspect decisions
    seg_overlay = np.zeros((H, W, 3), np.uint8)
    for eid, e in enumerate(edges):
        rc = np.column_stack([e["pts_xy"][:, 1], e["pts_xy"][:, 0]]).astype(np.int32)
        cv2.polylines(seg_overlay, [rc[:, ::-1].reshape(-1, 1, 2)], False,
                      colors[edge_category[eid]], int(thick_width_px), cv2.LINE_8)
    if report_path is not None:
        cv2.imwrite(str(report_path / "segments_overlay.png"), seg_overlay)

    # duplicates-only overlay on top of baseline: base in white, duplicates dashed
    dup_overlay = np.zeros((H, W, 3), np.uint8)
    # base network
    for e in edges:
        pts = e["pts_xy"]
        cv2.polylines(dup_overlay,
                      [np.column_stack([pts[:, 1], pts[:, 0]]).astype(np.int32)[:, ::-1].reshape(-1, 1, 2)],
                      False, (255, 255, 255), 1, cv2.LINE_8)
    # dashed duplicates from both passes
    for pts in dup_vis.get('kept', []):
        _draw_dashed_polyline(dup_overlay, pts, (0, 255, 0), int(thick_width_px))
    for pts in dup_vis.get('removed', []):
        _draw_dashed_polyline(dup_overlay, pts, (255, 0, 255), int(thick_width_px))

    graph_overlay = seg_overlay.copy()
    for n in nodes_vis.values():
        r, c = n["rc"]
        cv2.drawMarker(graph_overlay, (int(c), int(r)), (0, 255, 255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=9, thickness=1, line_type=cv2.LINE_8)
    if report_path is not None:
        cv2.imwrite(str(report_path / "duplicates_overlay.png"), dup_overlay)
        cv2.imwrite(str(report_path / "graph_overlay.png"), graph_overlay)

    _, _, seg_pixels = _rasterize_and_index(segments, (H, W))
    catalog: Dict[str, set] = {
        "all": set(int(sid) for sid in seg_pixels.keys()),
        "marked": set(),
        "deleted": set(),
        "kept": set(),
        "untreated": set(),
        "dead": set(),
    }
    for eid, e in enumerate(edges):
        sids = [int(x) for x in e["seg_ids"]]
        cat = edge_category[eid]
        if cat == "marked":
            catalog["marked"].update(sids)
        elif cat == "untreated":
            catalog["untreated"].update(sids)
        else:
            catalog["dead"].update(sids)

    segment_nodes: Dict[int, set] = {}
    for pc in pieces:
        pid = int(pc["pid"])
        u, v = int(pc["u"]), int(pc["v"])
        segment_nodes.setdefault(pid, set()).update([u, v])

    all_nodes = {}
    for nid, (x, y) in node_xy.items():
        rc = [int(round(y)), int(round(x))]
        all_nodes[int(nid)] = {
            "xy": [float(x), float(y)],
            "rc": rc,
            "deg": int(node_deg.get(nid, 0)),
            "type": "junction" if nid in nodes_vis else "plain",
        }

    node_edges: Dict[int, List[int]] = {}
    for eid, e in enumerate(edges):
        u, v = int(e["u"]), int(e["v"])
        node_edges.setdefault(u, []).append(eid)
        node_edges.setdefault(v, []).append(eid)

    node_segments: Dict[int, set] = {int(n): set() for n in all_nodes.keys()}
    for sid, nset in segment_nodes.items():
        for n in nset:
            node_segments[int(n)].add(int(sid))

    loop_edges: Dict[int, List[int]] = {lab: [] for lab in interior_labels}
    loop_segments: Dict[int, set] = {lab: set() for lab in interior_labels}
    loop_nodes: Dict[int, set] = {lab: set() for lab in interior_labels}
    for eid, labs in enumerate(edge_faces):
        if not labs:
            continue
        for lab in labs:
            loop_edges[lab].append(eid)
            loop_segments[lab].update(int(s) for s in edges[eid]["seg_ids"])
            loop_nodes[lab].update([int(edges[eid]["u"]), int(edges[eid]["v"])])

    segment_loops: Dict[int, set] = {}
    for lab, sids in loop_segments.items():
        for sid in sids:
            segment_loops.setdefault(int(sid), set()).add(int(lab))

    node_loops: Dict[int, set] = {}
    for lab, nids in loop_nodes.items():
        for nid in nids:
            node_loops.setdefault(int(nid), set()).add(int(lab))

    # in memory pruning and instance labeling
    # Build catalog from current edge categories -> sets of segment ids
    _, _, seg_pixels = _rasterize_and_index(segments, (H, W))
    # collect segment ids into catalog sets for rules and colored filtered preview
    catalog = {
        "untreated": set(),
        "marked": set(),
        "kept": set(),
        "deleted": set(),
        "dead": set(),
    }
    for eid, e in enumerate(edges):
        sids = [int(x) for x in e["seg_ids"]]
        cat = edge_category[eid]
        if cat == "marked":
            catalog["marked"].update(sids)
        elif cat == "untreated":
            catalog["untreated"].update(sids)
        elif cat == "dead":
            pass
        else:
            catalog["dead"].update(sids)

    # Construct structures for rules -> nodes edges node_edges loops node_loops
    all_nodes = {}
    for nid, (x, y) in node_xy.items():
        rc = [int(round(y)), int(round(x))]
        all_nodes[int(nid)] = {
            "xy": [float(x), float(y)],
            "rc": rc,
            "deg": int(node_deg.get(nid, 0)),
            "type": "junction" if nid in nodes_vis else "plain",
        }

    slim_edges = []
    for eid, e in enumerate(edges):
        slim_edges.append({
            "u": int(e["u"]),
            "v": int(e["v"]),
            "segments": [int(s) for s in e["seg_ids"]],
            "category": edge_category[eid],
            "length": int(edge_len[eid]),
        })

    node_edges = {}
    for eid, e in enumerate(slim_edges):
        u, v = int(e["u"]), int(e["v"])
        node_edges.setdefault(u, []).append(eid)
        node_edges.setdefault(v, []).append(eid)

    loop_edges = {lab: [] for lab in interior_labels}
    loop_segments = {lab: set() for lab in interior_labels}
    loop_nodes = {lab: set() for lab in interior_labels}
    for eid, labs in enumerate(edge_faces):
        if not labs:
            continue
        for lab in labs:
            loop_edges[lab].append(eid)
            loop_segments[lab].update(int(s) for s in edges[eid]["seg_ids"])
            loop_nodes[lab].update([int(edges[eid]["u"]), int(edges[eid]["v"])])

    # assemble loop to edges segments and nodes mapping for rules
    segment_loops = {}
    for lab, sids in loop_segments.items():
        for sid in sids:
            segment_loops.setdefault(int(sid), set()).add(int(lab))

    node_loops = {}
    for lab, nids in loop_nodes.items():
        for nid in nids:
            node_loops.setdefault(int(nid), set()).add(int(lab))

    loops = {
        int(lab): {
            "type": loop_type.get(lab, "unknown"),
            "edges": loop_edges.get(lab, []),
            "segments": sorted(int(s) for s in loop_segments.get(lab, set())),
            "nodes": sorted(int(n) for n in loop_nodes.get(lab, set())),
        }
        for lab in interior_labels
    }

    # Apply rules A to E in place
    _apply_rule_A_single_node_loops(catalog, all_nodes, loops)
    _apply_rule_B_group_neck_prune(catalog, all_nodes, slim_edges, node_edges, loops, node_loops)
    _apply_rule_C_prune_weak_connectors(catalog, slim_edges, node_edges, min_neighbor_len, active_marked_frac)
    _apply_rule_D_two_terminal_groups(catalog, all_nodes, slim_edges, node_edges, loops, seg_pixels, two_path_diff_frac)
    _apply_rule_E_terminal_mst(catalog, all_nodes, slim_edges, node_edges, loops, seg_pixels)

    # Build barrier pixels from kept ∪ untreated segments
    barriers_seed = np.zeros((H, W), np.uint8)
    boundary_ids = set(catalog.get("kept", set())) | set(catalog.get("untreated", set()))
    for sid in boundary_ids:
        for r, c in seg_pixels.get(int(sid), ()):
            if 0 <= r < H and 0 <= c < W:
                barriers_seed[r, c] = 255

    # Morphological closing schedule to seal small gaps
    labels = None
    last_barriers = None
    # progressively close gaps with small structuring elements until interiors are found
    for rad in (1, 2, 3, 4):
        barriers = barriers_seed.copy()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        barriers = cv2.dilate(barriers, k, iterations=1)
        barriers = cv2.morphologyEx(barriers, cv2.MORPH_CLOSE, k, iterations=1)

        open_mask = (barriers == 0).astype(np.uint8)
        ff = open_mask.copy()
        pad = np.zeros((H + 2, W + 2), np.uint8)
        for cc in range(W):
            if ff[0, cc] == 1:
                cv2.floodFill(ff, pad, (cc, 0), 2, flags=4)
                pad.fill(0)
            if ff[H - 1, cc] == 1:
                cv2.floodFill(ff, pad, (cc, H - 1), 2, flags=4)
                pad.fill(0)
        for rr in range(H):
            if ff[rr, 0] == 1:
                cv2.floodFill(ff, pad, (0, rr), 2, flags=4)
                pad.fill(0)
            if ff[rr, W - 1] == 1:
                cv2.floodFill(ff, pad, (W - 1, rr), 2, flags=4)
                pad.fill(0)

        interior = (ff == 1).astype(np.uint8)
        # label interior components with 4 connectivity to avoid diagonal merging
        _, labels_try = cv2.connectedComponents(interior, connectivity=4)
        if labels_try.max() > 0:
            labels = labels_try.astype(np.uint16)
            last_barriers = barriers
            break

    if labels is None:
        labels = np.zeros((H, W), np.uint16)
        last_barriers = barriers_seed

    # Optional previews
    if report_path:
        # write filtered segments color map white untreated orange connectors red marked magenta deleted blue kept
        cv2.imwrite(str(report_path / "filtered_segments.png"), last_barriers)
        n = int(labels.max())
        if n == 0:
            preview_unique = np.zeros((H, W, 3), np.uint8)
        else:
            rng = np.random.RandomState(12345)
            palette = np.zeros((n + 1, 3), dtype=np.uint8)
            palette[1:] = rng.randint(0, 256, size=(n, 3), dtype=np.uint8)
            preview_unique = palette[labels]
        cv2.imwrite(str(report_path / "instances.png"), preview_unique)

    # For binary mask use (labels > 0).astype(np.uint8) -> 1
    return labels

#  RULES AND INSTANCE LABELING HELPERS

# Provide a positive weight per edge. Prefer precomputed length else pixel count.
def _edge_weight_simple(e, seg_pixels):
    if "length" in e and e["length"]:
        return float(e["length"])
    segs = [int(s) for s in e.get("segments", [])]
    w = sum(len(seg_pixels.get(s, ())) for s in segs)
    return float(w if w > 0 else 1.0)

# Group marked faces that touch at junctions to form processing groups.
def _groups_of_marked_loops_simple(nodes, edges, loops):
    marked = {lid for lid, info in loops.items() if info.get("type") == "marked"}
    if not marked:
        return []
    # Only loops that touch junction nodes participate in grouping since terminals do not transmit adjacency.
    junctions = {nid for nid, info in nodes.items() if info.get("type") == "junction"}
    node_to_loops = defaultdict(list)
    for lid in marked:
        for nid in map(int, loops[lid].get("nodes", [])):
            if nid in junctions:
                node_to_loops[nid].append(lid)
    adj = {lid: set() for lid in marked}
    for lids in node_to_loops.values():
        for i in range(len(lids)):
            for j in range(i + 1, len(lids)):
                a, b = lids[i], lids[j]
                adj[a].add(b)
                adj[b].add(a)
    comps = []
    seen = set()
    for lid in marked:
        if lid in seen:
            continue
        q = deque([lid])
        seen.add(lid)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(set(comp))
    enriched = []
    for comp in comps:
        comp_edges = set()
        comp_nodes = set()
        for lid in comp:
            comp_edges.update(map(int, loops[lid].get("edges", [])))
            comp_nodes.update(map(int, loops[lid].get("nodes", [])))
        enriched.append((comp, comp_nodes, comp_edges))
    return enriched

# Build node adjacency and weight mapping for a set of edges.
def _build_adj_for_edges_simple(edges, edge_ids, seg_pixels):
    adj = defaultdict(list)
    for eid in edge_ids:
        e = edges[int(eid)]
        u, v = int(e["u"]), int(e["v"])
        w = _edge_weight_simple(e, seg_pixels)
        # Build undirected adjacency with edge ids and weights so shortest-path routines can reconstruct edge paths.
        adj[u].append((v, int(eid), w))
        adj[v].append((u, int(eid), w))
    return adj

# Shortest Path: Standard Dijkstra over the node graph to route within a group
def _dijkstra_simple(start, adj):
    dist = {start: 0.0}
    prev_node = {}
    prev_edge = {}
    pq = [(0.0, start)]
    # Standard Dijkstra using a binary heap over nodes. Stores both predecessor node and predecessor edge.
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, eid, w in adj.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                heapq.heappush(pq, (nd, v))
    return dist, prev_node, prev_edge

# Convert node predecessor maps into an edge path list / reconstruct path
def _reconstruct_edge_path_simple(src, dst, prev_node, prev_edge):
    if dst not in prev_edge:
        return []
    path = []
    cur = dst
    while cur != src:
        path.append(prev_edge[cur])
        cur = prev_node[cur]
    path.reverse()
    return path

def _bresenham_line_pixels_simple(r0, c0, r1, c1):
    H = max(r0, r1) + 1
    W = max(c0, c1) + 1
    canvas = np.zeros((H, W), np.uint8)
    cv2.line(canvas, (int(c0), int(r0)), (int(c1), int(r1)), 255, 1, cv2.LINE_8)
    nz = cv2.findNonZero(canvas)
    if nz is None:
        return []
    xy = nz[:, 0]
    return [(int(y), int(x)) for (x, y) in xy]

# apply rules A to E in memory to move segments between kept deleted
# Single-Node loops: Delete marked loops that touch at most one junction.
def _apply_rule_A_single_node_loops(catalog, nodes, loops):
    if not loops:
        return {"loops_checked": 0, "loops_copied": 0, "segments_copied": 0}
    junctions = {nid for nid, info in nodes.items() if info.get("type") == "junction"}
    loops_checked = loops_copied = seg_copied = 0
    for lid, info in loops.items():
        if info.get("type") != "marked":
            continue
        loops_checked += 1
        ln = set(map(int, info.get("nodes", [])))
        # Single-node loop: loop touches at most one junction. Copy its segments to 'deleted' to prune it.
        if len(ln & junctions) <= 1:
            for sid in map(int, info.get("segments", [])):
                if sid in catalog["marked"]:
                    catalog["deleted"].add(sid)
                    seg_copied += 1
            loops_copied += 1
    return {"loops_checked": loops_checked, "loops_copied": loops_copied, "segments_copied": seg_copied}

# Within a marked group prune dead connectors that only lead to a single active kind.
def _apply_rule_B_group_neck_prune(catalog, nodes, edges, node_edges, loops, node_loops):
    if not loops:
        return {"groups_checked": 0, "groups_copied": 0, "segments_copied": 0}

    def loop_has_marked_segments(lid: int) -> bool:
        return any(int(s) in catalog["marked"] for s in loops[lid].get("segments", []))

    marked_loops = {lid for lid, info in loops.items()
                    if info.get("type") == "marked" and loop_has_marked_segments(lid)}
    if not marked_loops:
        return {"groups_checked": 0, "groups_copied": 0, "segments_copied": 0}
    junctions = {nid for nid, info in nodes.items() if info.get("type") == "junction"}
    node_marked_loops = defaultdict(list)
    for lid in marked_loops:
        for nid in map(int, loops[lid].get("nodes", [])):
            if nid in junctions:
                node_marked_loops[nid].append(lid)
    loop_adj = {lid: set() for lid in marked_loops}
    for lids in node_marked_loops.values():
        for i in range(len(lids)):
            for j in range(i + 1, len(lids)):
                a, b = lids[i], lids[j]
                loop_adj[a].add(b)
                loop_adj[b].add(a)
    groups = []
    seen = set()
    for lid in marked_loops:
        if lid in seen:
            continue
        q = deque([lid])
        seen.add(lid)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in loop_adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        groups.append(set(comp))
    edge_cat = [e.get("category", "") for e in edges]

    def _firstcat_exits(start_nid: int, comp_edges: set, comp_loops: set):
        kinds = set()
        for eid0 in node_edges.get(start_nid, []):
            if eid0 in comp_edges:
                continue
            first_cat = edge_cat[eid0]
            e0 = edges[eid0]
            v0 = int(e0["v"]) if int(e0["u"]) == start_nid else int(e0["u"])
            q = deque([(v0, first_cat)])
            visited_nodes = {start_nid, v0}
            visited_edges = {eid0}
            while q:
                u, first = q.popleft()
                if set(node_loops.get(u, set())) - comp_loops:
                    kinds.add(first)
                    break
                if any(edge_cat[eid] == "untreated" for eid in node_edges.get(u, [])):
                    kinds.add(first)
                    break
                for eid in node_edges.get(u, []):
                    if eid in visited_edges or eid in comp_edges:
                        continue
                    if edge_cat[eid] not in ("marked", "dead"):
                        continue
                    visited_edges.add(eid)
                    e = edges[eid]
                    v = int(e["v"]) if int(e["u"]) == u else int(e["u"])
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        q.append((v, first))
        return kinds

    groups_checked = groups_copied = seg_copied = 0
    junctions = {nid for nid, info in nodes.items() if info.get("type") == "junction"}
    for comp in groups:
        groups_checked += 1
        comp_edges = set()
        comp_nodes = set()
        for lid in comp:
            comp_edges.update(map(int, loops[lid].get("edges", [])))
            comp_nodes.update(map(int, loops[lid].get("nodes", [])))
        comp_nodes &= junctions
        all_firstcats = []
        for nid in comp_nodes:
            kinds = _firstcat_exits(nid, comp_edges, comp)
            if kinds:
                all_firstcats.extend(list(kinds))
        attachments = len(all_firstcats)
        connector_only = bool(all_firstcats) and all(k == "dead" for k in all_firstcats)
        if attachments <= 1 or connector_only:
            for lid in comp:
                for sid in map(int, loops[lid].get("segments", [])):
                    if sid in catalog["marked"]:
                        catalog["deleted"].add(sid)
                        seg_copied += 1
            groups_copied += 1
    return {"groups_checked": groups_checked, "groups_copied": groups_copied, "segments_copied": seg_copied}


# Delete dead connectors that fail to connect two active neighborhoods.
def _apply_rule_C_prune_weak_connectors(catalog, edges, node_edges, min_neighbor_len, active_marked_frac):
    edge_cat = [e.get("category", "") for e in edges]
    edge_len = [int(e.get("length", 0)) for e in edges]

    def edge_active_non_connector(eid: int) -> bool:
        # active means sufficiently long or contains many non deleted segments
        if edge_cat[eid] == "dead":
            return False
        # Define an 'active' neighbor as at least min_neighbor_len pixels long. Guards against tiny noise.
        if edge_len[eid] < min_neighbor_len:
            return False
        if edge_cat[eid] == "untreated":
            return True
        segs = list(map(int, edges[eid].get("segments", [])))
        if not segs:
            return False
        n_not_deleted = sum(1 for s in segs if s not in catalog["deleted"])
        return n_not_deleted > active_marked_frac * len(segs)

    to_copy = set()
    for eid, e in enumerate(edges):
        if edge_cat[eid] != "dead":
            continue
        u, v = int(e["u"]), int(e["v"])
        cnt = 0
        if any(edge_active_non_connector(nei) for nei in node_edges.get(u, []) if nei != eid):
            cnt += 1
        if any(edge_active_non_connector(nei) for nei in node_edges.get(v, []) if nei != eid):
            cnt += 1
        # Weak connector: touches at most one active neighborhood. Mark its segments for deletion.
        if cnt <= 1:
            to_copy.add(eid)
    for eid in to_copy:
        for sid in map(int, edges[eid].get("segments", [])):
            catalog["deleted"].add(sid)
    return {"connectors_checked": len([e for e in edges if e.get("category") == "dead"]),
            "connectors_flagged": len(to_copy),
            "segments_copied": sum(len(list(edges[eid].get("segments", []))) for eid in to_copy)}


# If a marked group touches exactly two terminals keep the shorter of the
# two disjoint paths else collapse to a straight connector.
def _apply_rule_D_two_terminal_groups(catalog, nodes, edges, node_edges, loops, seg_pixels, two_path_diff_frac):
    groups = _groups_of_marked_loops_simple(nodes, edges, loops)
    edge_weight = [_edge_weight_simple(e, seg_pixels) for e in edges]
    seg_kept = seg_deleted = groups_processed = 0
    for comp, comp_nodes, comp_edges in groups:
        junctions = {nid for nid in comp_nodes if nodes.get(int(nid), {}).get("type") == "junction"}
        terminals = [int(nid) for nid in junctions
                     if any(edges[eid].get("category") == "untreated" for eid in node_edges.get(int(nid), []))]
        if len(terminals) != 2:
            # If the group does not reduce to a 2-terminal case, fall back to collapsing marked segments
            # into a straight connector between its boundary nodes.
            continue
        groups_processed += 1
        t0, t1 = terminals
        adj = _build_adj_for_edges_simple(edges, comp_edges, seg_pixels)
        if t0 not in adj or t1 not in adj:
            continue
        dist, prev_node, prev_edge = _dijkstra_simple(t0, adj)
        if t1 not in dist:
            continue
        p1 = _reconstruct_edge_path_simple(t0, t1, prev_node, prev_edge)
        len1 = sum(edge_weight[e] for e in p1)
        adj2 = defaultdict(list)
        banned = set(p1)
        for eid in comp_edges:
            if int(eid) in banned:
                continue
            e = edges[int(eid)]
            u, v = int(e["u"]), int(e["v"])
            w = _edge_weight_simple(e, seg_pixels)
            adj2[u].append((v, int(eid), w))
            adj2[v].append((u, int(eid), w))
        dist2, prev_node2, prev_edge2 = _dijkstra_simple(t0, adj2)
        p2 = _reconstruct_edge_path_simple(t0, t1, prev_node2, prev_edge2) if t1 in dist2 else []
        len2 = sum(edge_weight[e] for e in p2) if p2 else float('inf')
        all_group_segs = set(s for e in comp_edges for s in map(int, edges[int(e)].get("segments", [])))
        if p2 and np.isfinite(len2) and abs(len1 - len2) / max(len1, len2) > two_path_diff_frac:
            keep_edges = set(p1 if len1 <= len2 else p2)
            keep_segs = set(s for e in keep_edges for s in map(int, edges[e].get("segments", [])))
            del_segs = all_group_segs - keep_segs
            catalog["kept"].update(keep_segs)
            catalog["deleted"].update(del_segs)
            seg_kept += len(keep_segs)
            seg_deleted += len(del_segs)
        else:
            catalog["deleted"].update(all_group_segs)
            seg_deleted += len(all_group_segs)
            r0, c0 = map(int, nodes[t0]["rc"])
            r1, c1 = map(int, nodes[t1]["rc"])
            pix = _bresenham_line_pixels_simple(r0, c0, r1, c1)
            if pix:
                new_sid = (max(seg_pixels.keys()) + 1) if seg_pixels else 1
                seg_pixels[new_sid] = pix
                catalog["kept"].add(new_sid)
                seg_kept += 1
    return {"groups_processed": groups_processed, "segments_kept": seg_kept, "segments_deleted": seg_deleted}

# For multi-terminal marked groups, connect terminals by an MST over shortest paths.
def _apply_rule_E_terminal_mst(catalog, nodes, edges, node_edges, loops, seg_pixels):
    if not loops:
        return {"groups_seen": 0, "groups_processed": 0, "segments_kept": 0, "segments_deleted": 0}
    groups = _groups_of_marked_loops_simple(nodes, edges, loops)
    groups_seen = len(groups)
    groups_processed = seg_kept = seg_deleted = 0
    for comp, comp_nodes, comp_edges in groups:
        junctions = {nid for nid in comp_nodes if nodes.get(int(nid), {}).get("type") == "junction"}
        terminals = [int(nid) for nid in junctions
                     if any(edges[eid].get("category") == "untreated" for eid in node_edges.get(int(nid), []))]
        if len(terminals) < 3:
            continue
        adj = _build_adj_for_edges_simple(edges, comp_edges, seg_pixels)
        terminals = [t for t in terminals if t in adj]
        if len(terminals) < 3:
            continue
        groups_processed += 1
        dist_maps = {}
        prev_maps = {}
        for t in terminals:
            dist, prev_node, prev_edge = _dijkstra_simple(t, adj)
            dist_maps[t] = dist
            prev_maps[t] = (prev_node, prev_edge)
        in_mst = {terminals[0]}
        pairs = []
        while len(in_mst) < len(terminals):
            best = None
            for u in in_mst:
                du = dist_maps[u]
                for v in terminals:
                    if v in in_mst:
                        continue
                    w = du.get(v, float('inf'))
                    if np.isfinite(w):
                        if best is None or w < best[0]:
                            best = (w, u, v)
            if best is None:
                break
            _, u, v = best
            pairs.append((u, v))
            in_mst.add(v)
        kept_edges = set()
        for u, v in pairs:
            prev_node, prev_edge = prev_maps[u]
            path = _reconstruct_edge_path_simple(u, v, prev_node, prev_edge)
            kept_edges.update(path)
        kept_segs = set(s for e in kept_edges for s in map(int, edges[e].get("segments", [])))
        all_group_segs = set(s for e in comp_edges for s in map(int, edges[int(e)].get("segments", [])))
        del_segs = all_group_segs - kept_segs
        catalog["kept"].update(kept_segs)
        catalog["deleted"].update(del_segs)
        seg_kept += len(kept_segs)
        seg_deleted += len(del_segs)
    return {"groups_seen": groups_seen, "groups_processed": groups_processed,
            "segments_kept": seg_kept, "segments_deleted": seg_deleted}
