import numpy as np
import cv2
from typing import List, Tuple, Optional
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
    for lb in labels:
        out[mask == int(lb)] = colors[int(lb)]
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
    Convert segments into closed-loop instances. Save a debug overlay of loops to be removed.
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
        r0, c0 = seg[0];
        r1, c1 = seg[-1]
        is_close = (r0 - r1) ** 2 + (c0 - c1) ** 2 <= 2
        cv2.polylines(wall, [pts_xy], isClosed=(len(seg) >= 4 and is_close), color=255, thickness=1,
                      lineType=cv2.LINE_8)
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

    # Debug overlay: segment-based drawing of removed loops
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
            remap[lbl] = np.uint16(nid);
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
    Debug-only visualization. Segment-driven overlay.
    Draw ONLY wall segments that belong to loops slated for removal.
    Removal per label: area < min_area_px OR no central vein enclosed.
    Border-touching components are ignored. Portal-vessel check intentionally skipped.
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

    # Removal set
    removed_ids: List[int] = []
    for i in range(1, num_labels):
        if i in border_ids:
            continue
        region_bool = (lab == i)
        if not np.any(region_bool):
            continue
        area = int(areas[i])
        has_cv = bool(np.any(cv_mask[region_bool]))
        if area < max(1, int(min_area_px)) or (not has_cv):
            removed_ids.append(i)

    # Early out
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if not removed_ids:
        Path(report_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(report_path) / "filtered_loops.png"), canvas)
        return

    # Build rings for each removed label
    ring_dilate = 2
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    se_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_dilate + 1, 2 * ring_dilate + 1))
    rings = {}
    bboxes = {}
    for rid in removed_ids:
        region_u8 = (lab == rid).astype(np.uint8) * 255
        ring = cv2.morphologyEx(region_u8, cv2.MORPH_GRADIENT, se1)
        ring = cv2.dilate(ring, se_d, iterations=1)
        rings[rid] = ring
        ys, xs = np.where(ring > 0)
        if ys.size == 0:
            bboxes[rid] = (0, 0, -1, -1)
        else:
            bboxes[rid] = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

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
        rmin = int(arr[:, 0].min());
        rmax = int(arr[:, 0].max())
        cmin = int(arr[:, 1].min());
        cmax = int(arr[:, 1].max())
        seg_bboxes.append((rmin, cmin, rmax, cmax))

    def bbox_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)

    ring_overlap_frac = 0.15
    min_overlap_px = 3

    # Assign segments to removed loops and draw only those that *form the loop*
    snap_dist = 1.5  # px distance to ring to consider "on-boundary"
    ring_overlap_frac = 0.8
    min_overlap_px = 5

    for rid in removed_ids:
        rb = bboxes[rid]
        if rb[2] < rb[0] or rb[3] < rb[1]:
            continue
        ring = rings[rid]
        # Distance to ring: 0 at ring pixels
        dist = cv2.distanceTransform(255 - (ring > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3)

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

            # both endpoints on boundary and high overlap
            if L == 0:
                continue
            if not (on[0] and on[-1]):
                continue
            if hits < int(min_overlap_px) or hits / float(L) < float(ring_overlap_frac):
                continue

            seg = segments[sid]
            for (rA, cA), (rB, cB) in zip(seg[:-1], seg[1:]):
                cv2.line(canvas, (int(cA), int(rA)), (int(cB), int(rB)), (0, 0, 255), 1, lineType=cv2.LINE_8)

    Path(report_path).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(report_path) / "filtered_loops.png"), canvas)
