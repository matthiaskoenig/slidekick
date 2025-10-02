from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from pathlib import Path


def _colorize_labels(mask: np.ndarray) -> np.ndarray:
    """Colorize an instance-labeled mask for quick visualization."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return out
    rng = np.random.default_rng(12345)
    colors = {int(lb): rng.integers(64, 256, size=3, dtype=np.uint8) for lb in labels}
    # robust per-channel assignment avoids boolean indexing shape pitfalls
    for lb in labels:
        m = (mask == int(lb))
        clr = colors[int(lb)]
        out[..., 0][m] = clr[0]
        out[..., 1][m] = clr[1]
        out[..., 2][m] = clr[2]
    return out


def process_segments_to_mask(
        segments: List[List[Tuple[int, int]]],
        image_shape: Tuple[int, int],
        cv_contours: Optional[List[np.ndarray]] = None,
        portal_contours: Optional[List[np.ndarray]] = None,
        report_path: Optional[Path] = None,
        min_area_px: int = 50,
) -> np.ndarray:
    """
    Convert segments into closed-loop instances. Save a debug overlay.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    if H <= 0 or W <= 0:
        raise ValueError("image_shape must be positive (H, W)")

    # Rasterize 1px wall and thicken slightly
    wall = np.zeros((H, W), dtype=np.uint8)
    for seg in segments:
        if len(seg) < 2:
            continue
        pts_xy = np.array([[c, r] for (r, c) in seg], dtype=np.int32).reshape(-1, 1, 2)
        r0, c0 = seg[0]
        r1, c1 = seg[-1]
        is_close = (r0 - r1) ** 2 + (c0 - c1) ** 2 <= 2
        cv2.polylines(wall, [pts_xy], isClosed=(len(seg) >= 4 and is_close), color=255, thickness=1, lineType=cv2.LINE_8)
        for (rA, cA), (rB, cB) in zip(seg[:-1], seg[1:]):
            cv2.line(wall, (int(cA), int(rA)), (int(cB), int(rB)), color=255, thickness=1, lineType=cv2.LINE_8)

    wall_thin = wall.copy()
    if np.any(wall):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        wall = cv2.dilate(wall, k, iterations=1)

    if report_path is not None:
        Path(report_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(report_path) / "segments_raster.png"), wall)

    # Connected components on free space
    free = (wall == 0).astype(np.uint8)
    num, lab = cv2.connectedComponents(free, connectivity=8)

    # Debug overlay: all detected loop segments in red, kept paths in cyan, removed overlay in magenta
    loop_filtering(segments, lab, num, cv_contours, portal_contours, min_area_px, report_path, wall_thin)

    if num <= 1:
        return np.zeros((H, W), dtype=np.uint16)

    # Keep components not touching border
    border_labels = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    keep_mask = np.ones(num, dtype=bool)
    keep_mask[0] = False
    keep_mask[border_labels] = False

    remap = np.zeros(num, dtype=np.uint16)
    nid = 1
    for lbl in range(1, num):
        if keep_mask[lbl]:
            remap[lbl] = np.uint16(nid)
            nid += 1
    instance = remap[lab]

    if report_path is not None:
        cv2.imwrite(str(Path(report_path) / "polygon_mask.png"), _colorize_labels(instance))
        np.save(str(Path(report_path) / "polygon_mask_labels.npy"), instance)

    return instance


def loop_filtering(
        segments: List[List[Tuple[int, int]]],
        lab: np.ndarray,
        num: int,
        cv_contours: Optional[List[np.ndarray]],
        portal_contours: Optional[List[np.ndarray]],
        min_area_px: int,
        report_path: Optional[Path],
        wall_thin: Optional[np.ndarray] = None,
) -> None:
    """
    Visualization only.
    Red: all loop-forming segments.
    Cyan: new kept paths.
    Magenta: overlay for loops that get removed.
    """
    if report_path is None:
        return

    H, W = lab.shape[:2]

    # Central-vein mask from contours
    cv_mask = np.zeros((H, W), dtype=np.uint8)
    if cv_contours:
        try:
            cv2.drawContours(cv_mask, cv_contours, -1, 255, thickness=cv2.FILLED)
        except Exception:
            cv_cnts = [np.asarray(c, dtype=np.int32) for c in cv_contours]
            cv2.drawContours(cv_mask, cv_cnts, -1, 255, thickness=cv2.FILLED)

    # Identify border-touching labels
    border_ids = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))

    # Areas per label
    num_labels = int(lab.max()) + 1 if num is None else int(num)
    areas = np.bincount(lab.ravel(), minlength=num_labels)

    # Inner component ids
    inner_ids = [i for i in range(1, num_labels) if (i not in border_ids and np.any(lab == i))]

    # Removal set
    removed_ids: List[int] = []
    for i in inner_ids:
        region_bool = (lab == i)
        area = int(areas[i])
        has_cv = bool(np.any(cv_mask[region_bool]))
        if area < max(1, int(min_area_px)) or (not has_cv):
            removed_ids.append(i)

    # Prepare overlay
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Build rings and regions for all inner labels
    ring_dilate = 2
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    se_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_dilate + 1, 2 * ring_dilate + 1))

    rings: Dict[int, np.ndarray] = {}
    ring_dists: Dict[int, np.ndarray] = {}
    regions: Dict[int, np.ndarray] = {}
    bboxes: Dict[int, Tuple[int, int, int, int]] = {}

    for cid in inner_ids:
        region_u8 = (lab == cid).astype(np.uint8) * 255
        ring = cv2.morphologyEx(region_u8, cv2.MORPH_GRADIENT, se1)
        ring = cv2.dilate(ring, se_d, iterations=1)
        rings[cid] = ring
        ring_dists[cid] = cv2.distanceTransform(255 - (ring > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3)
        regions[cid] = region_u8
        ys, xs = np.where(ring > 0)
        if ys.size == 0:
            bboxes[cid] = (0, 0, -1, -1)
        else:
            bboxes[cid] = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

    # Precompute segment arrays and bboxes
    seg_coords: List[Optional[np.ndarray]] = []
    seg_bboxes: List[Tuple[int, int, int, int]] = []
    for seg in segments:
        if not seg or len(seg) < 2:
            seg_coords.append(None)
            seg_bboxes.append((0, 0, -1, -1))
            continue
        arr = np.array(seg, dtype=np.int32)
        seg_coords.append(arr)
        rmin = int(arr[:, 0].min())
        rmax = int(arr[:, 0].max())
        cmin = int(arr[:, 1].min())
        cmax = int(arr[:, 1].max())
        seg_bboxes.append((rmin, cmin, rmax, cmax))

    def bbox_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)

    # Parameters
    snap_dist = 1.5  # px distance to consider "on-boundary"
    ring_overlap_frac = 0.80
    min_overlap_px = 5

    # Helper: cluster points within tol (Chebyshev radius in px)
    def _cluster_points(pts: List[Tuple[int, int]], tol: int = 2) -> List[Tuple[int, int]]:
        if len(pts) <= 1:
            return pts
        pts_arr = np.array(pts, dtype=np.int32)
        used = np.zeros(len(pts), dtype=bool)
        centers: List[Tuple[int, int]] = []
        for i in range(len(pts)):
            if used[i]:
                continue
            pi = pts_arr[i]
            d = np.max(np.abs(pts_arr - pi), axis=1)  # Chebyshev
            group = np.where((d <= tol) & (~used))[0]
            used[group] = True
            cen = np.mean(pts_arr[group], axis=0).astype(int)
            centers.append((int(cen[0]), int(cen[1])))
        return centers

    # Helper: draw path along contour indices i..j going forward (cyan)
    def _draw_arc(cnt_xy: np.ndarray, i: int, j: int) -> None:
        N = len(cnt_xy)
        if N < 2:
            return
        idxs = []
        k = i
        while True:
            idxs.append(k)
            if k == j:
                break
            k = (k + 1) % N
        for a, b in zip(idxs[:-1], idxs[1:]):
            x1, y1 = int(cnt_xy[a, 0]), int(cnt_xy[a, 1])
            x2, y2 = int(cnt_xy[b, 0]), int(cnt_xy[b, 1])
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1, lineType=cv2.LINE_8)

    # Draw loop-forming segments for all inner labels in red. Overlay magenta where removed.
    connectors_by_rid: Dict[int, List[Dict]] = {rid: [] for rid in removed_ids}

    for cid in inner_ids:
        rb = bboxes[cid]
        if rb[2] < rb[0] or rb[3] < rb[1]:
            continue
        dist = ring_dists[cid]

        for sid, arr in enumerate(seg_coords):
            if arr is None:
                continue
            if not bbox_intersect(seg_bboxes[sid], rb):
                continue

            rs = arr[:, 0].clip(0, H - 1)
            cs = arr[:, 1].clip(0, W - 1)
            d = dist[rs, cs]
            on = d <= snap_dist
            L = int(len(arr))
            hits = int(on.sum())
            if L == 0:
                continue

            # Segment on loop boundary -> draw red baseline
            if on[0] and on[-1] and hits >= min_overlap_px and (hits / float(L)) >= ring_overlap_frac:
                seg = segments[sid]
                for (rA, cA), (rB, cB) in zip(seg[:-1], seg[1:]):
                    # red (BGR)
                    cv2.line(canvas, (int(cA), int(rA)), (int(cB), int(rB)), (0, 0, 255), 1, lineType=cv2.LINE_8)
                    # overlay magenta if this loop is slated for removal
                    if cid in removed_ids:
                        cv2.line(canvas, (int(cA), int(rA)), (int(cB), int(rB)), (255, 0, 255), 1, lineType=cv2.LINE_8)
                continue

            # Connector: exactly one endpoint on the ring
            if cid in removed_ids and (on[0] ^ on[-1]):
                near_end = 0 if on[0] else -1
                far_end = -1 if on[0] else 0
                node_rc = (int(arr[near_end, 0]), int(arr[near_end, 1]))
                far_rc = (int(arr[far_end, 0]), int(arr[far_end, 1]))

                # classify whether the far end is on any removed ring
                to_removed = False
                for rid2 in removed_ids:
                    dist2 = ring_dists[rid2]
                    if dist2[min(max(far_rc[0], 0), H - 1), min(max(far_rc[1], 0), W - 1)] <= snap_dist:
                        to_removed = True
                        break

                connectors_by_rid[cid].append({
                    "segment_id": int(sid),
                    "node": node_rc,
                    "far": far_rc,
                    "to_removed": bool(to_removed),
                })

    # For each removed loop, compute cyan kept path using the two-node rule
    for rid in removed_ids:
        region_u8 = regions[rid]

        # Nodes = unique connector touch points
        nodes_raw = [d["node"] for d in connectors_by_rid[rid]]
        node_pts = _cluster_points(nodes_raw, tol=2)

        if len(node_pts) < 2:
            continue

        if len(node_pts) == 2:
            # Extract outer contour for arc measurement
            cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            areas_cnt = [cv2.contourArea(c) for c in cnts]
            cnt = cnts[int(np.argmax(areas_cnt))]
            pts_xy = cnt[:, 0, :]  # (N,2) x,y

            def nearest_idx(node_rc: Tuple[int, int]) -> int:
                r, c = node_rc
                dx = pts_xy[:, 0] - c
                dy = pts_xy[:, 1] - r
                return int(np.argmin(dx * dx + dy * dy))

            i0 = nearest_idx(node_pts[0])
            i1 = nearest_idx(node_pts[1])

            dif = np.diff(pts_xy, axis=0, append=pts_xy[:1, :])
            edge = np.sqrt((dif[:, 0] ** 2 + dif[:, 1] ** 2).astype(np.float64))
            perim = float(edge.sum())

            def arc_len_forward(a: int, b: int) -> float:
                if a == b:
                    return 0.0
                if b >= a:
                    return float(edge[a:b].sum())
                else:
                    return float(edge[a:].sum() + edge[:b].sum())

            L1 = arc_len_forward(i0, i1)
            L2 = max(perim - L1, 0.0)

            if max(L1, L2) > 1.10 * min(L1, L2):
                if L1 <= L2:
                    _draw_arc(pts_xy, i0, i1)
                else:
                    _draw_arc(pts_xy, i1, i0)
            else:
                p0 = (int(pts_xy[i0, 0]), int(pts_xy[i0, 1]))
                p1 = (int(pts_xy[i1, 0]), int(pts_xy[i1, 1]))
                cv2.line(canvas, p0, p1, (255, 255, 0), 1, lineType=cv2.LINE_8)

    # Save overlay
    Path(report_path).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(report_path) / "filtered_loops.png"), canvas)
