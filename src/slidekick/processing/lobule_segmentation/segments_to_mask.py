# segments_to_mask.py
import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path


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


def _rasterize_skeleton(segments: Optional[List[np.ndarray]], shape: Tuple[int, int]) -> np.ndarray:
    H, W = map(int, shape)
    sk = np.zeros((H, W), dtype=np.uint8)
    if not segments:
        return sk
    for seg in segments:
        if seg is None or len(seg) < 2:
            continue
        pts = np.asarray(seg, dtype=float)
        for i in range(len(pts) - 1):
            r0, c0 = pts[i]
            r1, c1 = pts[i + 1]
            cv2.line(sk, (int(c0), int(r0)), (int(c1), int(r1)), 255, 1, lineType=cv2.LINE_8)
    return sk


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


def process_segments_to_mask(
    segments: Optional[List[np.ndarray]],
    image_shape: Tuple[int, int],
    cv_contours: Optional[List[np.ndarray]] = None,
    report_path: Optional[str] = None,
    min_area_px: int = 50,
    L_MIN: Optional[int] = None,
    node_scope: str = "removed",
) -> np.ndarray:
    H, W = map(int, image_shape)

    base_skeleton = _rasterize_skeleton(segments, (H, W))
    base_skeleton = (base_skeleton > 0).astype(np.uint8)

    num_holes, holes_lab = _holes_from_closed_skeleton(base_skeleton, close_iter=1)

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

    thin_bool = base_skeleton.astype(bool)
    deg = _degree_map(base_skeleton)

    side = min(H, W)
    L_MIN_auto = max(6, int(round(0.0025 * side))) if L_MIN is None else int(L_MIN)
    SHORT_LEN = max(3, int(round(0.0010 * side)))
    L_VIS = max(L_MIN_auto, SHORT_LEN)

    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k3 = np.ones((3, 3), dtype=np.uint8)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[thin_bool] = (150, 150, 150)

    if node_scope == "removed":
        loop_ids = removed_ids
    elif node_scope == "kept":
        loop_ids = [i for i in range(1, num_holes + 1) if keep_label[i]]
    else:
        loop_ids = list(range(1, num_holes + 1))

    node_points: List[Tuple[int, int]] = []

    # Precompute ring for all loops to allow stopping BFS when touching any loop
    all_rings = np.zeros((H, W), dtype=np.uint8)
    for hid in range(1, num_holes + 1):
        region_i = (holes_lab == hid).astype(np.uint8) * 255
        ring_i = cv2.morphologyEx(region_i, cv2.MORPH_GRADIENT, se3)
        all_rings |= (ring_i > 0).astype(np.uint8)

    for hid in loop_ids:
        region = (holes_lab == hid).astype(np.uint8) * 255
        if region.sum() == 0:
            continue

        ring_thin = cv2.morphologyEx(region, cv2.MORPH_GRADIENT, se3)
        near_ring = cv2.dilate(ring_thin, se3, iterations=1) > 0
        red_mask = thin_bool & (ring_thin > 0)

        canvas[(ring_thin > 0)] = (0, 0, 255)

        junction = (deg >= 3)
        walkable = thin_bool & (~near_ring)
        walkable_n = cv2.filter2D(walkable.astype(np.uint8), cv2.CV_16S, k3, borderType=cv2.BORDER_CONSTANT)

        cand = junction & near_ring & (walkable_n > 0)
        diag_touch = (deg == 2) & near_ring & (walkable_n > 0)
        cand |= diag_touch

        # connectivity seeds: walkable pixels that touch near_ring
        touch_near = cv2.filter2D((near_ring > 0).astype(np.uint8), cv2.CV_16S, k3, borderType=cv2.BORDER_CONSTANT) > 0
        seeds = walkable & touch_near

        # BFS from seeds limited by L_VIS and stopping when hitting any loop ring
        visited = np.zeros((H, W), dtype=np.uint8)
        sr, sc = np.where(seeds)
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

            q = [(rr2, cc2, 1) for rr2, cc2 in nbrs]
            seen = set([(r, c)] + [(rr2, cc2) for rr2, cc2 in nbrs])
            best_d = 0
            accept = False

            while q and not accept:
                pr, pc, d = q.pop(0)
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
                    q.append((nr, nc, d + 1))

            if accept or best_d >= SHORT_LEN:
                node_points.append((r, c))

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
                thickness=2,
                line_type=cv2.LINE_8,
            )

    if report_path is not None:
        cv2.imwrite(str(Path(report_path) / "filtered_loops.png"), canvas)

    if num_holes == 0:
        return np.zeros((H, W), dtype=np.uint16)
    remap = np.zeros(num_holes + 1, dtype=np.uint16)
    for hid in range(1, num_holes + 1):
        remap[hid] = hid
    instance = remap[holes_lab].astype(np.uint16)
    return instance
