import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json


RING_SAMPLE_STEP = 3  # subsample contour points when mapping loop->segments (>=1). 1 = no subsampling.


def _neighbors8(r: int, c: int, H: int, W: int):
    r0 = max(0, r - 1)
    r1 = min(H - 1, r + 1)
    c0 = max(0, c - 1)
    c1 = min(W - 1, c + 1)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr == r and cc == c:
                continue
            yield rr, cc


def _degree_map(skel01: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=np.uint8)
    conv = cv2.filter2D(skel01.astype(np.uint8), ddepth=cv2.CV_16S, kernel=k, borderType=cv2.BORDER_CONSTANT)
    deg = (conv - skel01.astype(np.int16)).astype(np.int16)
    return deg


def _holes_from_closed_skeleton(skel01: np.ndarray, close_iter: int = 1) -> Tuple[int, np.ndarray]:
    if close_iter > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = (skel01.astype(np.uint8) * 255).copy()
        for _ in range(close_iter):
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, se, iterations=1)
    else:
        closed = (skel01.astype(np.uint8) * 255).copy()

    bg = (closed == 0).astype(np.uint8)
    num_labels, lab = cv2.connectedComponents(bg, connectivity=4)

    H, W = lab.shape
    border_ids = set(np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    keep = np.ones(num_labels, dtype=bool)
    for bid in border_ids:
        keep[bid] = False
    keep[0] = False

    remap = np.zeros(num_labels, dtype=np.int32)
    nid = 1
    for i in range(1, num_labels):
        if keep[i]:
            remap[i] = nid
            nid += 1
    holes_lab = remap[lab]
    num_holes = nid - 1
    return num_holes, holes_lab


def _rasterize_and_index(
    segments: Optional[List[np.ndarray]],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = map(int, shape)
    skel01 = np.zeros((H, W), dtype=np.uint8)
    S_idx = np.full((H, W), -1, dtype=np.int32)
    if not segments:
        return skel01, S_idx

    tmp = np.zeros((H, W), dtype=np.uint8)

    for sid, seg in enumerate(segments):
        if seg is None or len(seg) < 2:
            continue
        pts = np.asarray(seg, dtype=np.float32)
        pts = np.round(pts).astype(np.int32)  # (row, col)
        if len(pts) < 2:
            continue

        tmp.fill(0)
        pts_cv = pts[:, ::-1].reshape(-1, 1, 2)  # (x,y)
        cv2.polylines(tmp, [pts_cv], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_8)

        nz = cv2.findNonZero(tmp)
        if nz is None:
            continue
        xy = nz[:, 0]
        cc = xy[:, 0]
        rr = xy[:, 1]

        skel01[rr, cc] = 1
        mnew = (S_idx[rr, cc] == -1)
        if np.any(mnew):
            S_idx[rr[mnew], cc[mnew]] = sid

    return skel01, S_idx


def _ordered_loop_segments_voronoi(
    ring_thin: np.ndarray,
    labels: np.ndarray,
    S_idx: np.ndarray,
    W: int,
    sample_step: int = 1,
) -> List[int]:
    contours, _ = cv2.findContours(ring_thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)  # (x,y)

    if sample_step > 1:
        pts = pts[::sample_step]

    H, Wimg = labels.shape
    assert Wimg == W

    ids: List[int] = []
    for x, y in pts:
        y = int(y)
        x = int(x)
        if not (0 <= y < H and 0 <= x < W):
            continue
        lbl = int(labels[y, x])
        if lbl <= 0:
            sid = int(S_idx[y, x])
        else:
            idx0 = lbl - 1
            y0 = idx0 // W
            x0 = idx0 % W
            sid = int(S_idx[y0, x0])
        ids.append(sid)

    ordered: List[int] = []
    last = None
    for sid in ids:
        if sid == -1:
            continue
        if last is None or sid != last:
            ordered.append(sid)
            last = sid
    if len(ordered) >= 2 and ordered[0] == ordered[-1]:
        ordered.pop()
    return ordered


def process_segments_to_mask(
    segments: Optional[List[np.ndarray]],
    image_shape: Tuple[int, int],
    cv_contours: Optional[List[np.ndarray]] = None,
    report_path: Optional[str] = None,
    min_area_px: int = 50,
    L_MIN: Optional[int] = None,
    node_scope: str = "removed",
) -> np.ndarray:
    """
    Debug overlay and hole-instance label map.
      grey   = original skeleton
      red    = segments of loops to be treated
      magenta= treated segments slated for deletion
      cyan   = treated segments kept
      white  = segments of other loops kept as is
      orange = dead branches after deletion (disabled here)
    """
    H, W = map(int, image_shape)

    base_skeleton01, S_idx = _rasterize_and_index(segments, (H, W))
    thin_bool = base_skeleton01.astype(bool)

    num_holes, holes_lab = _holes_from_closed_skeleton(base_skeleton01, close_iter=1)

    cv_mask = np.zeros((H, W), dtype=np.uint8)
    if cv_contours:
        try:
            cv2.drawContours(cv_mask, cv_contours, contourIdx=-1, color=255, thickness=cv2.FILLED, lineType=cv2.LINE_8)
        except Exception:
            cv_cnts = [np.asarray(cnt, dtype=np.int32) for cnt in cv_contours]
            cv2.drawContours(cv_mask, cv_cnts, contourIdx=-1, color=255, thickness=cv2.FILLED, lineType=cv2.LINE_8)

    keep_label = np.zeros(num_holes + 1, dtype=bool)
    removed_ids: List[int] = []
    for hid in range(1, num_holes + 1):
        region = (holes_lab == hid)
        area = int(region.sum())
        has_cv = bool((cv_mask[region] > 0).any())
        if area >= int(min_area_px) and has_cv:
            keep_label[hid] = True
        else:
            removed_ids.append(hid)

    deg = _degree_map(base_skeleton01)

    side = min(H, W)
    L_MIN_auto = max(6, int(round(0.0025 * side))) if L_MIN is None else int(L_MIN)
    SHORT_LEN = max(3, int(round(0.0010 * side)))
    L_VIS = max(L_MIN_auto, SHORT_LEN)

    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k3 = np.ones((3, 3), dtype=np.uint8)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[thin_bool] = (150, 150, 150)

    if node_scope == "removed":
        ring_scope_ids = removed_ids
    elif node_scope == "kept":
        ring_scope_ids = [i for i in range(1, num_holes + 1) if keep_label[i]]
    else:
        ring_scope_ids = list(range(1, num_holes + 1))

    all_rings = np.zeros((H, W), dtype=np.uint8)
    ring_cache: Dict[int, np.ndarray] = {}
    near_ring_cache: Dict[int, np.ndarray] = {}
    for hid in range(1, num_holes + 1):
        region_i = (holes_lab == hid).astype(np.uint8) * 255
        ring_i = cv2.morphologyEx(region_i, cv2.MORPH_GRADIENT, se3)
        ring_cache[hid] = ring_i
        near_ring_cache[hid] = cv2.dilate(ring_i, se3, iterations=1) > 0
        all_rings |= (ring_i > 0).astype(np.uint8)

    inv = np.where(base_skeleton01 > 0, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        inv, distanceType=cv2.DIST_L2, maskSize=3, labelType=cv2.DIST_LABEL_PIXEL
    )

    loop2seg: Dict[int, List[int]] = {}
    seg2loops: Dict[int, List[int]] = {}

    node_points: List[Tuple[int, int]] = []
    nodes_by_loop: Dict[int, List[Tuple[int, int]]] = {}

    for hid in range(1, num_holes + 1):
        ring_thin = ring_cache[hid]
        near_ring = near_ring_cache[hid]

        if hid in ring_scope_ids:
            ring_on_skel = thin_bool & near_ring
            canvas[ring_on_skel] = (0, 0, 255)

        ordered_seg_ids = _ordered_loop_segments_voronoi(
            ring_thin, labels, S_idx, W, sample_step=max(1, int(RING_SAMPLE_STEP))
        )
        loop2seg[hid] = ordered_seg_ids
        for sid in ordered_seg_ids:
            if sid < 0:
                continue
            lst = seg2loops.get(sid)
            if lst is None:
                seg2loops[sid] = [hid]
            else:
                if not lst or lst[-1] != hid:
                    lst.append(hid)

        if hid not in ring_scope_ids:
            continue

        junction = (deg >= 3)
        walkable = thin_bool & (~near_ring)
        walkable_n = cv2.filter2D(walkable.astype(np.uint8), cv2.CV_16S, k3, borderType=cv2.BORDER_CONSTANT)
        cand = junction & near_ring & (walkable_n > 0)
        diag_touch = (deg == 2) & near_ring & (walkable_n > 0)
        cand |= diag_touch

        touch_near = cv2.filter2D((near_ring > 0).astype(np.uint8), cv2.CV_16S, k3, borderType=cv2.BORDER_CONSTANT) > 0
        seeds = walkable & touch_near
        visited = np.zeros((H, W), dtype=np.uint8)
        sr, sc = np.where(seeds)
        if len(sr) > 0:
            q = [(int(r), int(c), 0) for r, c in zip(sr, sc)]
            for r, c, _ in q:
                visited[r, c] = 1
            while q:
                pr, pc, d = q.pop(0)
                if d >= L_VIS:
                    continue
                for nr, nc in _neighbors8(pr, pc, H, W):
                    if visited[nr, nc]:
                        continue
                    if not walkable[nr, nc]:
                        continue
                    if all_rings[nr, nc]:
                        continue
                    visited[nr, nc] = 1
                    q.append((nr, nc, d + 1))
            canvas[visited.astype(bool)] = (0, 0, 255)

        rr, cc = np.where(cand)
        for r, c in zip(rr, cc):
            nbrs = [(rr2, cc2) for rr2, cc2 in _neighbors8(r, c, H, W) if walkable[rr2, cc2]]
            if not nbrs:
                continue
            q2 = [(rr2, cc2, 1) for rr2, cc2 in nbrs]
            seen = set([(r, c)] + [(rr2, cc2) for rr2, cc2 in nbrs])
            best_d = 0
            accept = False
            while q2 and not accept:
                pr, pc, d = q2.pop(0)
                best_d = max(best_d, d)
                if d >= L_MIN_auto:
                    accept = True
                    break
                for nr, nc in _neighbors8(pr, pc, H, W):
                    if not walkable[nr, nc]:
                        continue
                    if (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    q2.append((nr, nc, d + 1))
            if accept or best_d >= SHORT_LEN:
                node_points.append((r, c))
                nodes_by_loop.setdefault(hid, []).append((r, c))

    if len(node_points) > 0:
        pts = np.array(node_points, dtype=np.int32)
        cluster_r = 3 if side >= 256 else 2
        used = np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            if used[i]:
                continue
            p = pts[i]
            dcheb = np.max(np.abs(pts - p), axis=1)
            grp = np.where((dcheb <= cluster_r) & (~used))[0]
            used[grp] = True
            r, c = np.mean(pts[grp], axis=0).astype(int)
            cv2.drawMarker(
                canvas,
                (int(c), int(r)),
                (0, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=9,
                thickness=1,
                line_type=cv2.LINE_8,
            )

    cluster_r = 3 if side >= 256 else 2
    cluster_counts: Dict[int, int] = {}
    for hid, pts_list in nodes_by_loop.items():
        if not pts_list:
            cluster_counts[hid] = 0
            continue
        arr = np.asarray(pts_list, dtype=np.int32)
        used = np.zeros(len(arr), dtype=bool)
        count = 0
        for i in range(len(arr)):
            if used[i]:
                continue
            p = arr[i]
            dcheb = np.max(np.abs(arr - p), axis=1)
            grp = np.where((dcheb <= cluster_r) & (~used))[0]
            used[grp] = True
            count += 1
        cluster_counts[hid] = count

    conn_counts: Dict[int, int] = {hid: len(loop2seg.get(hid, [])) for hid in ring_scope_ids}

    few_node_ids = [
        hid for hid in ring_scope_ids
        if (cluster_counts.get(hid, 0) < 2) and (conn_counts.get(hid, 0) < 2)
    ]
    for hid in few_node_ids:
        ring_on_skel = thin_bool & near_ring_cache[hid]
        canvas[ring_on_skel] = (255, 0, 255)

    catalog: Dict[str, set] = {
        "all": set(),
        "marked": set(),
        "deleted": set(),
        "kept": set(),
        "untreated": set(),
        "dead": set(),
    }

    if segments is not None:
        catalog["all"] = set(range(len(segments)))
    else:
        catalog["all"] = set(int(s) for s in np.unique(S_idx) if s >= 0)

    treated: set = set()
    for hid in ring_scope_ids:
        for sid in loop2seg.get(hid, []):
            if sid >= 0:
                treated.add(int(sid))
    catalog["marked"] = treated

    other_loop_ids = [hid for hid in range(1, num_holes + 1) if hid not in ring_scope_ids]
    other_loop_seg_ids: set = set()
    for hid in other_loop_ids:
        for sid in loop2seg.get(hid, []):
            if sid >= 0:
                other_loop_seg_ids.add(int(sid))

    catalog["untreated"] = other_loop_seg_ids - treated

    deleted: set = set()
    for hid in few_node_ids:
        for sid in loop2seg.get(hid, []):
            if sid >= 0:
                deleted.add(int(sid))
    catalog["deleted"] = deleted

    catalog["kept"] = set()

    def _paint(seg_ids: set, bgr: Tuple[int, int, int]):
        if not seg_ids:
            return
        m = np.isin(S_idx, np.fromiter(seg_ids, dtype=np.int32))
        canvas[m] = bgr

    _paint(catalog["untreated"], (255, 255, 255))    # white
    _paint(catalog["marked"], (0, 0, 255))          # red
    _paint(catalog["kept"], (255, 255, 0))           # cyan
    _paint(catalog["deleted"], (255, 0, 255))        # magenta
    _paint(catalog["dead"], (0, 165, 255))           # orange

    if report_path is not None:
        outdir = Path(report_path)
        outdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outdir / "filtered_loops.png"), canvas)

    if num_holes == 0:
        return np.zeros((H, W), dtype=np.uint16)
    remap = np.zeros(num_holes + 1, dtype=np.uint16)
    for hid in range(1, num_holes + 1):
        remap[hid] = hid
    instance = remap[holes_lab].astype(np.uint16)
    return instance
