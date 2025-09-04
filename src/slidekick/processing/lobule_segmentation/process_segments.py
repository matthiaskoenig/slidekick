import numpy as np
import cv2
from typing import List, Tuple
from pathlib import Path

def process_segments_to_mask(
    segments: List[List[Tuple[int, int]]],
    image_shape: Tuple[int, int],
    report_path: Path = None
) -> np.ndarray:
    """
    Build a labeled lobule mask from closed line segments.

    Args:
        segments: list of segments from segment_thinned_image(...)  (each: [(row, col), ...])
        image_shape: (H, W) of the original skeleton image; mask will match this size exactly.

    Returns:
        mask: int32 H×W where 0 = background, 1..N = filled polygon IDs (one per closed segment).
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((H, W), dtype=np.int32)

    label_id = 1
    for seg in segments:
        # keep only closed loops
        if not seg or seg[0] != seg[-1]:
            continue
        # need at least a triangle (distinct points)
        if len(seg) < 4:
            continue

        # convert (row, col) -> (x, y) = (col, row) for OpenCV
        poly = np.array([(c, r) for (r, c) in seg], dtype=np.int32)

        # Optional guard: ensure there are at least 3 unique vertices
        if len(np.unique(poly, axis=0)) < 3:
            continue

        # Fill polygon with unique label
        cv2.fillPoly(mask, [poly], color=int(label_id))
        label_id += 1

    if report_path is not None:
        cv2.imwrite(str(report_path / "polygon_mask.png"), mask)

    return mask
