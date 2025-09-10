import numpy as np
import cv2
from skimage.filters import threshold_multiotsu
from skimage.morphology import closing, disk
from skimage.color import label2rgb
from typing import Tuple, Dict
from pathlib import Path


def multiotsu_split(gray: np.ndarray, classes: int = 3, blur_sigma: float = 1.5, report_path: Path = None):
    """
    3-class Otsu on a blurred grayscale -> labels in {0,1,2}.
    Heuristic: lowest mean = true background, middle = microscope bg, highest = tissue.
    """
    g = gray.astype(np.uint8)
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    # thresholds length = classes-1 -> two thresholds for 3 classes
    thr = threshold_multiotsu(g, classes=classes)
    lbl = np.digitize(gray, bins=thr)  # uses original gray for sharper boundaries
    # order by class mean
    means = [gray[lbl == i].mean() if np.any(lbl == i) else -1 for i in range(classes)]
    order = np.argsort(means)  # low -> high
    idx_true_bg, idx_mic_bg, idx_tissue = order[0], order[1], order[-1]
    if report_path is not None:
        # Build single-channel mask with requested gray codes
        vis = np.zeros_like(g, dtype=np.uint8)
        # Map each semantic class to its grayscale value
        if np.any(lbl == idx_true_bg):
            vis[lbl == idx_true_bg] = 0  # black
        if np.any(lbl == idx_mic_bg):
            vis[lbl == idx_mic_bg] = 128  # mid gray
        if np.any(lbl == idx_tissue):
            vis[lbl == idx_tissue] = 255  # white

        # Create parent dir if needed; accept file or directory path
        rp = Path(report_path)
        if rp.is_dir():
            rp = rp / "multiotsu_mask.png"
        rp.parent.mkdir(parents=True, exist_ok=True)

        # Write as PNG; lossless and preserves 8-bit grayscale
        cv2.imwrite(str(rp), vis)


    return lbl.astype(np.int32), (idx_true_bg, idx_mic_bg, idx_tissue)


def detect_tissue_mask_multiotsu(gray: np.ndarray,
                                 morphological_radius: int = 5, report_path: Path = None):
    """
    Build boolean masks: tissue, microscope background, true background.
    Cleans with closing + small object/hole removal.
    """
    lbl, (i0, i1, i2) = multiotsu_split(gray, classes=3, blur_sigma=15, report_path=report_path)
    m_tis  = (lbl == i2)

    m_tis = closing(m_tis, disk(int(morphological_radius)))

    return m_tis.astype(bool)


def overlay_mask(image_stack: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay segmentation labels using skimage.color.label2rgb.
    - Treats label 0 as background.
    - Supports multi-class masks.
    """

    base = image_stack.mean(axis=2).astype(np.float32) / 255.0
    lbl = mask.astype(np.int32)

    over = label2rgb(lbl, image=base, bg_label=0, alpha=alpha, image_alpha=1.0)
    return (over * 255).astype(np.uint8)


def build_mask_pyramid_from_processed(
    mask_cropped: np.ndarray,
    img_size_base: Tuple[int, int],            # (Hb, Wb) cropped ROI at base_level (AFTER bbox crop, BEFORE padding)
    bbox_base: Tuple[int, int, int, int],      # (min_r, min_c, max_r, max_c) in base_level coords
    orig_shapes: Dict[int, Tuple[int, int]],   # {level: (H,W)} full-frame shapes at each level
    base_level: int,                           # the level used to load/crop
) -> Dict[int, np.ndarray]:
    """
    0) mask_cropped: mask with padding already stripped (so shape == processed ROI size without pad)
    1) Resize mask_cropped from processed-ROI size -> base-level ROI size (img_size_base)
    2) Paste into a full-size base_level canvas at bbox_base
    3) Resample that base canvas to every level in orig_shapes (NEAREST to preserve labels)
    Returns {level: full_mask_at_level}
    """
    # Step 1: processed ROI -> base-level ROI
    Hb, Wb = int(img_size_base[0]), int(img_size_base[1])      # target ROI size at base_level
    if Hb <= 0 or Wb <= 0 or mask_cropped.size == 0:
        return {lvl: np.zeros(orig_shapes[lvl], dtype=np.int32) for lvl in orig_shapes}

    roi_base = cv2.resize(
        mask_cropped.astype(np.int32),
        (Wb, Hb),  # (width, height)
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    # Step 2: paste into full-size base canvas at bbox
    Hfull_base, Wfull_base = orig_shapes[base_level]
    min_r, min_c, max_r, max_c = bbox_base
    canvas_base = np.zeros((Hfull_base, Wfull_base), dtype=np.int32)
    # Safety clamp (in case of off-by-one)
    min_r = max(0, min_r); min_c = max(0, min_c)
    max_r = min(Hfull_base, max_r); max_c = min(Wfull_base, max_c)
    if (max_r - min_r) != Hb or (max_c - min_c) != Wb:
        # If bbox dims and img_size_base mismatch by 1 px, reconcile by resize
        Hb2, Wb2 = (max_r - min_r), (max_c - min_c)
        roi_base = cv2.resize(roi_base, (Wb2, Hb2), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    canvas_base[min_r:max_r, min_c:max_c] = roi_base

    # Step 3: build pyramid by resizing base canvas to each level
    out = {}
    for lvl, (Hdst, Wdst) in orig_shapes.items():
        if lvl == base_level:
            out[lvl] = canvas_base.copy()
        else:
            out[lvl] = cv2.resize(
                canvas_base, (Wdst, Hdst), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
    return out

def pad_image(image_stack: np.ndarray, pad: int) -> np.ndarray:
    pad_width = ((pad, pad), (pad, pad), (0, 0))
    return np.pad(image_stack, pad_width, mode="constant", constant_values=0)