"""Stain normalization and separation."""
from typing import Tuple

import cv2
import numpy as np

from slidekick.processing.stain_separation import (
    calculate_stain_matrix,
    deconvolve_image_and_normalize,
    create_single_channel_pixels,
    reconstruct_pixels
)


def separate_stains(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    otsu_sample = gs.reshape(-1, 1)  # gs[mask[:]]

    threshold, _ = cv2.threshold(otsu_sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    idx = gs < threshold
    px_oi = image_array[idx]

    stain_matrix = calculate_stain_matrix(px_oi)

    concentrations = deconvolve_image_and_normalize(image_array.reshape(-1, 3), stain_matrix, maxCRef=None)

    h, e = create_single_channel_pixels(concentrations)

    return h.reshape(gs.shape), e.reshape(gs.shape)


def normalize_stain(
        image_array: np.ndarray,
        HERef: np.ndarray = np.array([[0.65, 0.2159], [0.70, 0.8012], [0.29, 0.5581]]),
        maxCRef: np.ndarray = np.array([1.9705, 1.0308])
) -> np.ndarray:
    """HE stain normalization.

    HERef = [[0.65, 0.2159], [0.70, 0.8012], [0.29, 0.5581]]
    maxCRef = [1.9705, 1.0308]

    :param image_array: image to normalize
    :param HERef: reference stain vectors
    :param maxCRef: maximal values for normalization
    :return:
    """

    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    otsu_sample = gs.reshape(-1, 1)  # gs[mask[:]]

    threshold, _ = cv2.threshold(otsu_sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    idx = gs < threshold
    px_oi = image_array[idx]

    stain_matrix = calculate_stain_matrix(px_oi)

    concentrations = deconvolve_image_and_normalize(image_array.reshape(-1, 3), stain_matrix, maxCRef)

    normalized = reconstruct_pixels(concentrations=concentrations, refrence_matrix=HERef)

    return normalized.reshape(image_array.shape)
