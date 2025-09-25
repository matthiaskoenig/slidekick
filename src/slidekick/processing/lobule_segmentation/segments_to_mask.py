import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path

# TODO: Dropped segments check if they can be mapped to outer lobule
# TODO: Then drop dead branches

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
    areas = np.bincount(lab_u.ravel(), minlength=num)
    tiny = areas < max(1, int(min_area_px))
    keep_mask[tiny] = False

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
        try:
            # PNG will clip to 8-bit; still useful for quick look. Prefer color preview.
            cv2.imwrite(str(report_path / "polygon_mask.png"), _colorize_labels(instance))
        except Exception:
            pass
        try:
            np.save(str(report_path / "polygon_mask_labels.npy"), instance)
        except Exception:
            pass

    return instance
