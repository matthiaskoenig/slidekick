import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
from collections import defaultdict, deque


def _save_small_loops_segments_debug(
        segments: List[List[Tuple[int, int]]],
        image_shape: Tuple[int, int],
        lab: np.ndarray,
        min_area_px: int,
        report_path: Optional[Path],
        ring_overlap_frac: float = 0.15,
        min_overlap_px: int = 3,
        ring_dilate: int = 2,
        endpoint_snap: int = 2,
        length_ratio_keep_shorter: float = 1.3,
) -> None:
    H, W = int(image_shape[0]), int(image_shape[1])
    num = int(lab.max()) + 1
    if num <= 1:
        return

    def bresenham(r0: int, c0: int, r1: int, c1: int):
        x0, y0, x1, y1 = int(c0), int(r0), int(c1), int(r1)
        pts = [];
        dx = abs(x1 - x0);
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1;
        sy = 1 if y0 < y1 else -1;
        err = dx + dy
        while True:
            pts.append((y0, x0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; x0 += sx
            if e2 <= dx: err += dx; y0 += sy
        return pts

    def skeleton_path(region_bool: np.ndarray, p0_rc, p1_rc):
        H2, W2 = region_bool.shape
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner = cv2.erode((region_bool.astype(np.uint8)) * 255, se)
        try:
            thinning = cv2.ximgproc.thinning
            skel = thinning(inner, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except Exception:
            skel = cv2.morphologyEx(inner, cv2.MORPH_GRADIENT, se)
        sk = skel > 0
        if sk.sum() == 0:
            return None
        coords = np.column_stack(np.where(sk))
        if coords.size == 0:
            return None

        def nearest(rc):
            d = (coords[:, 0] - rc[0]) ** 2 + (coords[:, 1] - rc[1]) ** 2
            return tuple(coords[int(np.argmin(d))])

        s0 = nearest((int(p0_rc[0]), int(p0_rc[1])))
        s1 = nearest((int(p1_rc[0]), int(p1_rc[1])))
        visited = np.full((H2, W2), False, dtype=bool);
        parent = {}
        q = deque([s0]);
        visited[s0] = True
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        found = False
        while q:
            r, c = q.popleft()
            if (r, c) == s1: found = True; break
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H2 and 0 <= cc < W2 and sk[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True;
                    parent[(rr, cc)] = (r, c);
                    q.append((rr, cc))
        if not found:
            return None
        path = [];
        cur = s1
        while cur != s0: path.append(cur); cur = parent[cur]
        path.append(s0);
        path.reverse()
        return [tuple(map(int, rc)) for rc in path]

    def closest_index(poly_xy: np.ndarray, pt_xy):
        if poly_xy is None or len(poly_xy) == 0: return -1
        px, py = int(pt_xy[0]), int(pt_xy[1])
        d = (poly_xy[:, 0] - px) ** 2 + (poly_xy[:, 1] - py) ** 2
        return int(np.argmin(d))

    def arc_lengths(poly_xy: np.ndarray, i: int, j: int):
        poly2 = np.vstack([poly_xy, poly_xy[0:1]])
        d = np.sqrt(np.sum(np.diff(poly2, axis=0) ** 2, axis=1))
        c = np.concatenate([[0], np.cumsum(d)])
        i0, j0 = sorted([i, j])
        L1 = float(c[j0] - c[i0]);
        total = float(c[-1]);
        L2 = total - L1
        return L1, L2

    def _axis_diag_score(arr: np.ndarray) -> float:
        if arr is None or len(arr) < 2: return 0.0
        dr = np.diff(arr[:, 0]);
        dc = np.diff(arr[:, 1])
        nz = (dr != 0) | (dc != 0)
        if not nz.any(): return 0.0
        dr = dr[nz];
        dc = dc[nz]
        hv = (dr == 0) | (dc == 0)
        diag = (np.abs(dr) == np.abs(dc)) & (dr != 0) & (dc != 0)
        return float((hv | diag).sum()) / float(len(dr))

    # small loop ids
    border_ids = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    areas = np.bincount(lab.ravel(), minlength=num)
    small_ids = [i for i in range(1, num) if areas[i] < max(1, int(min_area_px)) and i not in border_ids]

    # rings and bboxes
    rings = {};
    bboxes = {}
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    se_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2 * ring_dilate + 1, 2 * ring_dilate + 1)) if ring_dilate > 0 else None
    for rid in small_ids:
        region_u8 = (lab == rid).astype(np.uint8) * 255
        ring = cv2.morphologyEx(region_u8, cv2.MORPH_GRADIENT, se1)
        if ring_dilate > 0: ring = cv2.dilate(ring, se_d, iterations=1)
        rings[rid] = ring
        ys, xs = np.where(ring > 0)
        if ys.size == 0:
            bboxes[rid] = (0, 0, -1, -1)
        else:
            bboxes[rid] = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

    # segment arrays and bboxes
    seg_coords = [];
    seg_bboxes = []
    for seg in segments:
        if not seg or len(seg) < 2:
            seg_coords.append(None);
            seg_bboxes.append((0, 0, -1, -1));
            continue
        arr = np.array(seg, dtype=np.int32)
        seg_coords.append(arr)
        rmin = int(arr[:, 0].min());
        rmax = int(arr[:, 0].max())
        cmin = int(arr[:, 1].min());
        cmax = int(arr[:, 1].max())
        seg_bboxes.append((rmin, cmin, rmax, cmax))

    def bbox_intersect(a, b):
        ar0, ac0, ar1, ac1 = a;
        br0, bc0, br1, bc1 = b
        return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)

    # loop walls by sampling
    loop_segments = {rid: set() for rid in small_ids}
    seg_to_loops = {sid: set() for sid in range(len(segments))}
    for rid in small_ids:
        rb = bboxes[rid]
        if rb[2] < rb[0] or rb[3] < rb[1]: continue
        ring_bool = rings[rid] > 0
        for sid, arr in enumerate(seg_coords):
            if arr is None: continue
            if not bbox_intersect(seg_bboxes[sid], rb): continue
            rs = arr[:, 0].clip(0, H - 1);
            cs = arr[:, 1].clip(0, W - 1)
            hits = int(ring_bool[rs, cs].sum());
            L = int(len(arr))
            if hits >= int(min_overlap_px) or (L > 0 and hits / float(L) >= float(ring_overlap_frac)):
                loop_segments[rid].add(sid);
                seg_to_loops[sid].add(rid)

    shared_walls = {sid for sid, rids in seg_to_loops.items() if len(rids) >= 2}

    # snapped endpoint graph
    node_coords = []
    incident = defaultdict(set)
    seg_nodes = []

    def find_or_make_node(pt):
        r, c = int(pt[0]), int(pt[1])
        for nid, (rr, cc) in enumerate(node_coords):
            if abs(rr - r) <= endpoint_snap and abs(cc - c) <= endpoint_snap:
                return nid
        nid = len(node_coords);
        node_coords.append((r, c));
        return nid

    for sid, seg in enumerate(segments):
        if not seg or len(seg) < 2:
            seg_nodes.append((-1, -1));
            continue
        u = find_or_make_node(seg[0]);
        v = find_or_make_node(seg[-1])
        seg_nodes.append((u, v))
        incident[u].add(sid);
        incident[v].add(sid)

    # draw walls in red
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for rid in small_ids:
        for sid in loop_segments[rid]:
            seg = segments[sid]
            if len(seg) < 2: continue
            for (rA, cA), (rB, cB) in zip(seg[:-1], seg[1:]):
                cv2.line(canvas, (int(cA), int(rA)), (int(cB), int(rB)), (0, 0, 255), 1, lineType=cv2.LINE_8)

    # propose cyan reconnections
    for rid in small_ids:
        wall_set = loop_segments[rid]
        if not wall_set: continue

        wall_nodes = set()
        for sid in wall_set:
            u, v = seg_nodes[sid]
            if u >= 0: wall_nodes.add(u)
            if v >= 0: wall_nodes.add(v)

        # connectors: nodes where any incident seg is outside this loop's wall_set
        connectors = []
        for n in wall_nodes:
            inc = incident[n]
            if any((sid not in wall_set) for sid in inc):
                connectors.append(node_coords[n])

        k = len(connectors)
        if k <= 1:
            continue

        region_mask = (lab == rid).astype(np.uint8)
        cnts, _ = cv2.findContours(region_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        poly_xy = cnt[:, 0, :]

        if k == 2:
            n0, n1 = connectors[0], connectors[1]
            i0 = closest_index(poly_xy, (n0[1], n0[0]));
            i1 = closest_index(poly_xy, (n1[1], n1[0]))
            if i0 < 0 or i1 < 0:
                continue
            # Build both arcs between the connectors
            if i0 <= i1:
                arcA = poly_xy[i0:i1 + 1]
                arcB = np.vstack([poly_xy[i1:], poly_xy[:i0 + 1]])
            else:
                arcA = poly_xy[i1:i0 + 1]
                arcB = np.vstack([poly_xy[i0:], poly_xy[:i1 + 1]])

            # Orientation-dominant rule FIRST
            def segs_on_arc(arc_poly):
                m = np.zeros((H, W), dtype=np.uint8)
                for (x1, y1), (x2, y2) in zip(arc_poly[:-1], arc_poly[1:]):
                    cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1, lineType=cv2.LINE_8)
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
                keep = []
                for sid in wall_set:
                    arr = seg_coords[sid]
                    if arr is None:
                        continue
                    rs = arr[:, 0].clip(0, H - 1);
                    cs = arr[:, 1].clip(0, W - 1)
                    if int(m[rs, cs].sum()) >= int(min_overlap_px):
                        keep.append(arr)
                return keep

            A_score = max((_axis_diag_score(a) for a in segs_on_arc(arcA)), default=0.0)
            B_score = max((_axis_diag_score(b) for b in segs_on_arc(arcB)), default=0.0)
            if (A_score >= 0.6) ^ (B_score >= 0.6):
                keep_poly = arcB if A_score >= 0.6 else arcA
                for (x1, y1), (x2, y2) in zip(keep_poly[:-1], keep_poly[1:]):
                    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2, lineType=cv2.LINE_8)
                continue
            # Fallback to length ratio logic
            L1, L2 = arc_lengths(poly_xy, i0, i1)
            if max(L1, L2) <= length_ratio_keep_shorter * min(L1, L2):
                # Similar lengths: draw midline
                path = skeleton_path(region_mask > 0, n0, n1) or bresenham(n0[0], n0[1], n1[0], n1[1])
                for (rA, cA), (rB, cB) in zip(path[:-1], path[1:]):
                    cv2.line(canvas, (int(cA), int(rA)), (int(cB), int(rB)), (255, 255, 0), 2, lineType=cv2.LINE_8)
            else:
                # Unequal: keep shorter arc -> draw it
                if i0 <= i1:
                    arc_short = poly_xy[i0:i1 + 1] if L1 <= L2 else np.vstack([poly_xy[i1:], poly_xy[:i0 + 1]])
                else:
                    arc_short = poly_xy[i1:i0 + 1] if L1 <= L2 else np.vstack([poly_xy[i0:], poly_xy[:i1 + 1]])
                for (x1, y1), (x2, y2) in zip(arc_short[:-1], arc_short[1:]):
                    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2, lineType=cv2.LINE_8)

        else:
            # k>=3
            M = cv2.moments(region_mask.astype(np.uint8))
            if M['m00'] > 0:
                cy = int(round(M['m01'] / M['m00']));
                cx = int(round(M['m10'] / M['m00']))
            else:
                ys, xs = np.where(region_mask > 0);
                cy = int(round(ys.mean()));
                cx = int(round(xs.mean()))
            for (rN, cN) in connectors:
                path = bresenham(rN, cN, cy, cx)
                for (rA, cA), (rB, cB) in zip(path[:-1], path[1:]):
                    cv2.line(canvas, (int(cA), int(rA)), (int(cB), int(rB)), (255, 255, 0), 2, lineType=cv2.LINE_8)

            report_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(report_path / "small_loops.png"), canvas)


# TODO: Also filter CVs
# TODO: Replacement strategy for shared walls
# TODO: Then drop dead branches
# TODO: Then map changes to map


def _ensure_closed_polyline(seg: List[Tuple[int, int]], max_gap2: int = 2) -> Optional[np.ndarray]:
    """
    Ensure a segment is a closed polygonal chain.
    - seg is in (row, col). OpenCV wants (x=col, y=row).
    - If start/end are within sqrt(max_gap2) px, close it by repeating the start.
    - If still not closed or <3 vertices, return None.
    """
    if len(seg) < 3:
        return None
    r0, c0 = seg[0]
    r1, c1 = seg[-1]
    # close tiny gap
    if (r0 - r1) ** 2 + (c0 - c1) ** 2 <= max_gap2:
        seg = seg + [seg[0]]

    # valid polygon must start == end and have >= 4 points including closure
    if seg[0] != seg[-1] or len(seg) < 4:
        return None

    pts = np.array([[c, r] for (r, c) in seg], dtype=np.int32).reshape(-1, 1, 2)
    return pts


def _colorize_labels(mask: np.ndarray) -> np.ndarray:
    """
    Quick pseudo-color for a label mask (H,W), uint16 -> RGB uint8 for debugging.
    """
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return out
    # stable but distinct-ish colors
    rng = np.random.default_rng(12345)
    colors = {}
    for lb in labels:
        colors[int(lb)] = rng.integers(64, 256, size=3, dtype=np.uint8)
    for lb in labels:
        out[mask == lb] = colors[int(lb)]
    return out


def process_segments_to_mask(
        segments: List[List[Tuple[int, int]]],
        image_shape: Tuple[int, int],
        vessel_contours: Optional[List[List[Tuple[int, int]]]] = None,
        report_path: Optional[Path] = None,
        min_area_px: int = 50,
) -> np.ndarray:
    """
    ZIA-style processing of line segments into lobule instances + (new) instance mask.

    Pipeline (mirrors ZIA's semantics):
      1) Rasterize all segment polylines onto a binary "wall" image; slightly thicken to close tiny gaps.
      2) Find enclosed regions by labeling the complement of the wall image.
      3) Drop the outside/background component(s) touching borders.
      4) Remove tiny components (area < min_area_px).
      5) Compactly relabel remaining components to 1..N in a uint16 instance mask.
      6) Optionally carve out vessel contours as holes (set to 0) — default like ZIA.

    Returns
    -------
    np.ndarray
        (H, W) uint16 instance mask. Background=0; lobules=1..N.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    if H <= 0 or W <= 0:
        raise ValueError("image_shape must be positive (H, W)")

    # 1) Rasterize segment lines to "wall" image (uint8), then thicken
    wall = np.zeros((H, W), dtype=np.uint8)

    # Draw each segment as a polyline; if it looks closed, draw closed to help sealing.
    # We also draw line-by-line to be robust to repeated points.
    for seg in segments:
        if len(seg) < 2:
            continue
        # Convert to (x,y) = (col,row)
        pts_xy = np.array([[c, r] for (r, c) in seg], dtype=np.int32).reshape(-1, 1, 2)

        # Decide closure: cheap proximity check on endpoints
        r0, c0 = seg[0]
        r1, c1 = seg[-1]
        is_close = (r0 - r1) ** 2 + (c0 - c1) ** 2 <= 2

        if len(seg) >= 4 and is_close:
            # closed polyline
            cv2.polylines(wall, [pts_xy], isClosed=True, color=255, thickness=1, lineType=cv2.LINE_8)
        else:
            # open polyline (still draw as is)
            cv2.polylines(wall, [pts_xy], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_8)

        # additionally connect consecutive points in case of duplicates/gaps
        for (rA, cA), (rB, cB) in zip(seg[:-1], seg[1:]):
            cv2.line(wall, (int(cA), int(rA)), (int(cB), int(rB)), color=255, thickness=1, lineType=cv2.LINE_8)

    # Morphological dilation to close 1–2 px gaps between edges
    # (ZIA implicitly ensures small connectivity; we emulate with a tiny kernel)
    if np.any(wall):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        wall = cv2.dilate(wall, k, iterations=1)

    if report_path is not None:
        report_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(report_path / "segments_raster.png"), wall)

    # 2) Label the COMPLEMENT (free space) of the walls
    #    - inside lobules == connected components in (~wall)
    free = (wall == 0).astype(np.uint8)  # 1 where free, 0 where wall
    # OpenCV connectedComponents requires 8-connectivity for natural interiors
    num, lab = cv2.connectedComponents(free, connectivity=8)
    # lab: 0..num-1, where 0 is background per OpenCV (but our 'free' uses 1=free)
    _save_small_loops_segments_debug(segments, (H, W), lab, min_area_px, report_path)

    if num <= 1:
        # No free-space components other than background -> return empty
        mask_empty = np.zeros((H, W), dtype=np.uint16)
        if report_path is not None:
            cv2.imwrite(str(report_path / "polygon_mask.png"), mask_empty)
        return mask_empty

    # 3) Drop outside/background components that touch the image border
    lab_u = lab  # int32
    keep_mask = np.ones(num, dtype=bool)  # which component IDs to keep; will turn off border-touchers

    # Find labels on borders
    border_labels = np.unique(
        np.concatenate([
            lab_u[0, :], lab_u[-1, :], lab_u[:, 0], lab_u[:, -1]
        ])
    )
    # Drop any component that touches the border
    keep_mask[border_labels] = False

    # 4) Remove tiny components by area threshold
    # Compute areas for all labels
    # areas = np.bincount(lab_u.ravel(), minlength=num)
    # tiny = areas < max(1, int(min_area_px))
    # keep_mask[tiny] = False

    # Build a mapping old_label -> new_label (1..N), others -> 0
    remap = np.zeros(num, dtype=np.uint16)
    next_id = 1
    for lbl in range(1, num):  # skip 0; but note: 0 can be free-space "background" component
        if keep_mask[lbl]:
            remap[lbl] = np.uint16(next_id)
            next_id += 1

    instance = remap[lab_u]  # (H,W) uint16

    # If no kept components, return empty
    if next_id == 1:
        if report_path is not None:
            cv2.imwrite(str(report_path / "polygon_mask.png"), instance)
        return instance

    # 5) Optionally carve out vessel holes (set to 0 inside vessel contours)
    """
    if vessel_contours:
        # Draw each vessel polygon as filled 0 over instance labels
        for v in vessel_contours:
            if v is None or len(v) < 3:
                continue
            # close polygon if needed
            v_closed = _ensure_closed_polyline(v, max_gap2=4)
            if v_closed is None:
                # still allow filling with an open contour by closing via polylines isClosed=True
                pts_xy = np.array([[c, r] for (r, c) in v], dtype=np.int32).reshape(-1, 1, 2)
                # draw onto a temp mask to create fill region
                tmp = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(tmp, [pts_xy], 255, lineType=cv2.LINE_8)
                instance[tmp > 0] = 0
                continue
            cv2.fillPoly(instance, [v_closed], 0, lineType=cv2.LINE_8)
    """
    # 6) (Optional) relabel compactly again after holes (keeps 1..N contiguous)
    if vessel_contours:
        # Recompute connected components over non-zero labels, but preserve instances:
        # Safer: just compact labels by order of unique labels appearance
        uniq = np.unique(instance)
        uniq = uniq[uniq != 0]
        if uniq.size > 0:
            remap2 = {int(u): i + 1 for i, u in enumerate(uniq.tolist())}
            # vectorized remap
            flat = instance.ravel()
            out = np.zeros_like(flat, dtype=np.uint16)
            # create LUT-like mapping
            for old, new in remap2.items():
                out[flat == old] = np.uint16(new)
            instance = out.reshape(instance.shape)

    if report_path is not None:
        # Save raw label image (uint16) and a color preview
        # PNG will clip to 8-bit; still useful for quick look. Prefer color preview.
        cv2.imwrite(str(report_path / "polygon_mask.png"), _colorize_labels(instance))

        np.save(str(report_path / "polygon_mask_labels.npy"), instance)

    return instance
