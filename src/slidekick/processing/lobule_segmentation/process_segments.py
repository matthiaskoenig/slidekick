import numpy as np
import cv2
from typing import List, Tuple
from pathlib import Path

def process_segments_to_mask(
    segments: List[List[Tuple[int, int]]],
    image_shape: Tuple[int, int],
    vessel_contours: List[List[Tuple[int, int]]] = None,
    report_path: Path = None
) -> np.ndarray:
    """
    Build a labeled lobule mask from closed line segments, removing small polygons
    and carving out enclosed vessel contours as holes.

    Args:
        segments: List of closed skeleton segments (each a list of (row, col) points).
        image_shape: (H, W) tuple for the size of the output mask (matches the skeleton image).
        vessel_contours: Optional list of closed vessel contour segments to be treated as holes
                         (each: [(row, col), ...] with first point == last point).
        report_path: Optional Path to a directory; if provided, saves a debug image "polygon_mask.png".

    Returns:
        mask: int32 H×W array where 0 = background, 1..N = filled lobule IDs (holes for vessels are 0).
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((H, W), dtype=np.int32)

    # Define area threshold to filter out very small polygons (noise)
    MIN_AREA_PX = 50  # adjust threshold as needed based on image scale

    label_id = 1
    # Fill each lobule segment as a separate label
    for seg in segments:
        # Only consider closed loops with sufficient points (at least a triangle)
        if not seg or seg[0] != seg[-1]:
            continue
        if len(seg) < 4:
            continue

        # Convert (row, col) -> (x, y) = (col, row) for OpenCV fillPoly
        poly = np.array([(c, r) for (r, c) in seg], dtype=np.int32)
        # Ensure there are at least 3 unique vertices forming a valid polygon
        if len(np.unique(poly, axis=0)) < 3:
            continue

        # Skip extremely small polygons based on area threshold (noise filter)
        area = abs(cv2.contourArea(poly))
        if area < MIN_AREA_PX:
            continue

        # Fill the polygon on the mask with a new label
        cv2.fillPoly(mask, [poly], color=int(label_id))
        label_id += 1

    # If vessel contours are provided, carve them out as holes inside lobules
    if vessel_contours is not None:
        for vessel in vessel_contours:
            if not vessel or vessel[0] != vessel[-1]:
                continue
            if len(vessel) < 4:
                continue

            poly_v = np.array([(c, r) for (r, c) in vessel], dtype=np.int32)
            if len(np.unique(poly_v, axis=0)) < 3:
                continue

            area_v = abs(cv2.contourArea(poly_v))
            if area_v < MIN_AREA_PX:
                continue

            # Check if this vessel contour lies inside an existing lobule region
            M = cv2.moments(poly_v)
            if M["m00"] == 0:
                # Degenerate contour (no area)
                continue
            # Use the contour's centroid as a test point for enclosure
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if 0 <= cy < H and 0 <= cx < W and mask[cy, cx] != 0:
                # If the vessel's centroid is inside a labeled lobule, fill the vessel area with 0 (hole)
                cv2.fillPoly(mask, [poly_v], color=0)
            # If mask[cy, cx] == 0, the vessel is not geometrically enclosed in any lobule – do nothing

    # Save debug mask image if report_path is specified
    if report_path is not None:
        cv2.imwrite(str(report_path / "polygon_mask.png"), mask)

    return mask
