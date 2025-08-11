from typing import Optional, Tuple

import numpy as np
import cv2

# copied from https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

def calculate_stain_matrix(pxi: np.ndarray, Io=240, alpha=1) -> np.ndarray:
    """
    calculates the stain base vectors
    @param pxi: pixels of interest as np array of RGB tuples with shape(h*w, 3)
    @param Io: transmitted light intensity
    @param alpha: percentile to find robust maxima
    """

    od = -np.log((pxi.astype(float) + 1) / Io)

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(od.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    t_hat = od.dot(eigvecs[:, 1:3])

    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v_min = eigvecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eigvecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if v_min[0] > v_max[0]:
        stain_matrix = np.array((v_min[:, 0], v_max[:, 0])).T
    else:
        stain_matrix = np.array((v_max[:, 0], v_min[:, 0])).T

    return stain_matrix


def find_max_c(pxi, stain_matrix: np.ndarray, Io=240) -> np.ndarray:
    C = lsq(pxi, stain_matrix, Io)

    # normalize stain concentrations
    return np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

"""
def lsq(pxi: np.ndarray, stain_matrix: np.ndarray, Io=240) -> np.ndarray:
    y = -np.log((pxi.astype(float) + 1) / Io).T

    # determine concentrations of the individual stains
    return np.linalg.lstsq(stain_matrix, y, rcond=None)[0]
"""

# New func due to memory issues
def lsq(pxi: np.ndarray, stain_matrix: np.ndarray, Io=240) -> np.ndarray:
    """
    Memory-efficient least-squares for stain deconvolution.

    - Uses the pseudo-inverse of the small `stain_matrix` (computed once).
    - Operates in float32.
    - If `pxi` contains many pixels, it processes them in chunks to keep peak memory low.
    - Returns concentrations with shape (n_stains, n_pixels) to match previous behavior.

    Parameters
    ----------
    pxi : np.ndarray
        Pixel array. Expected shapes:
          - (n_pixels, 3)  -> common case (rows of RGB pixels)
          - (3, n_pixels)  -> already transposed case (will be handled)
    stain_matrix : np.ndarray
        Matrix of stain vectors (shape should be (3, n_stains) typically).
    Io : int or float
        Illumination constant used in OD transform (default 240).
    """
    # ensure numpy arrays and reduce precision to float32 to save memory
    A = np.asarray(stain_matrix, dtype=np.float32)
    p = np.asarray(pxi, dtype=np.float32, order='C')

    # compute pseudo-inverse of the small stain matrix (cheap)
    pinvA = np.linalg.pinv(A).astype(np.float32)  # shape: (n_stains, 3)

    # helper to compute concentrations for a 2D block of pixels with shape (m,3)
    def _process_block(block: np.ndarray) -> np.ndarray:
        # compute optical density: shape -> (3, m)
        od = -np.log((block.astype(np.float32) + 1.0) / float(Io)).T
        # concentrations: (n_stains, m)
        return pinvA.dot(od)

    # Case A: pxi is (n_pixels, 3)  -> typical
    if p.ndim == 2 and p.shape[1] == 3:
        n_pixels = p.shape[0]

        # choose a safe chunk size in pixels (tunable). This keeps memory peaks small.
        chunk_size = 200_000

        if n_pixels <= chunk_size:
            return _process_block(p)
        else:
            # new: preallocate and fill
            n_stains = pinvA.shape[0]
            out = np.empty((n_stains, n_pixels), dtype=np.float32)
            for i in range(0, n_pixels, chunk_size):
                blk = p[i:i + chunk_size]
                out[:, i:i + blk.shape[0]] = _process_block(blk)
            return out

    # Case B: pxi is already (3, n_pixels)
    if p.ndim == 2 and p.shape[0] == 3:
        # compute od directly (shape (3, n_pixels))
        od = -np.log((p.astype(np.float32) + 1.0) / float(Io))
        return pinvA.dot(od)

    # other shapes are unexpected -> provide helpful error
    raise ValueError(
        "lsq: unexpected pxi shape. Expected (n_pixels, 3) or (3, n_pixels). "
        f"Got shape {p.shape}."
    )

def deconvolve_image(pxi: np.ndarray, stain_matrix: np.ndarray, maxC: np.ndarray, maxCRef=None, Io=240) -> np.ndarray:
    """
    deconvolution of the stains for the pixels of interest.
    @param pxi: pixels of interest
    @param stain_matrix: m x 2 matrix where m is rgb channel for 2 stain colors
    @param Io: transmission intensity
    @return: Io, he, dab np arrays of shapes (m*n, 3), (m*n,), (m*n,)
    """

    # determine concentrations of the individual stains
    C = lsq(pxi, stain_matrix, Io)

    # we do not need those reference max concentrations. We don't know anyway
    if maxCRef is not None:
        tmp = np.divide(maxC, maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

    else:
        # That should actually contain the information about the cyps (concentration)
        # as given by Lambert Beer log(I0/I) = e*c*d where c is concentration
        # However, some pixels have negative concentrations
        C2 = np.divide(C, maxC[:, np.newaxis])

    # recreate the image using reference mixing matrix
    # Inorm = reconstruct_pixels(C2)

    # unmix hematoxylin and eosin
    # CHANGED: Instead of using reference matrix for mixing, the image is just
    # the concentration is just exponentiated into a single channel
    #

    return C2


def deconvolve_image_and_normalize(pxi: np.ndarray, stain_matrix: np.ndarray, maxCRef: Optional[np.ndarray], Io=240) -> np.ndarray:
    """
    deconvolution of the stains for the pixels of interest.
    @param pxi: pixels of interest
    @param stain_matrix: m x 2 matrix where m is rgb channel for 2 stain colors
    @param Io: transmission intensity
    @return: Io, he, dab np arrays of shapes (m*n, 3), (m*n,), (m*n,)
    """

    # determine concentrations of the individual stains
    C = lsq(pxi, stain_matrix, Io)

    # we do not need those reference max concentrations. We don't know anyway

    # That should actually contain the information about the cyps (concentration)
    # as given by Lambert Beer log(I0/I) = e*c*d where c is concentration
    # However, some pixels have negative concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

    if maxCRef is not None:
        tmp = np.divide(maxC, maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

    else:
        # That should actually contain the information about the cyps (concentration)
        # as given by Lambert Beer log(I0/I) = e*c*d where c is concentration
        # However, some pixels have negative concentrations
        C2 = np.divide(C, maxC[:, np.newaxis])

    return C2


def create_single_channel_pixels(concentrations: np.ndarray, Io=240) -> np.ndarray:
    """
    reconstructs the pixel values for one stain using the concentrations from
    the deconvolution
    @param concentrations: shape (m, ) matrix containing the concentration of the stain
    @param Io: transmission intensity
    @return:
    """

    i = np.multiply(Io, np.exp(-concentrations))
    i[i > 255] = 254

    return i.astype(np.uint8)


def reconstruct_pixels(concentrations: np.ndarray, refrence_matrix=None, Io=240):
    """
    reconstructs the image pixels using the reference_matrix and the concentrations
    from the deconvolution
    @param concentrations: shape (m, 2) matrix containg the concentrations of the stains
    @param refrence_matrix: shape (3,2) matrix containing RGB color vectors for the stains
    @param Io: transmission intensity
    @return:
    """

    HERef = np.array([[0, 1],
                      [1, 1],
                      [1, 0]])

    if refrence_matrix is not None:
        HERef = refrence_matrix

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(concentrations)))
    Inorm[Inorm > 255] = 254
    Inorm = Inorm.astype(np.uint8).T.reshape(-1, 3)
    return Inorm


def separate_stains(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    threshold, _ = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    idx = gs < threshold
    ##px_oi = image_array[idx]
    px_oi = image_array[idx][::10]  # subsampling for speed

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
    ##px_oi = image_array[idx]
    px_oi = image_array[idx][::10]  # subsampling for speed

    stain_matrix = calculate_stain_matrix(px_oi)

    concentrations = deconvolve_image_and_normalize(image_array.reshape(-1, 3).astype(np.float32, copy=False), stain_matrix, maxCRef)

    normalized = reconstruct_pixels(concentrations=concentrations, refrence_matrix=HERef)

    return normalized.reshape(image_array.shape)