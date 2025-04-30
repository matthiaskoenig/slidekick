from typing import Optional

import numpy as np


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


def lsq(pxi: np.ndarray, stain_matrix: np.ndarray, Io=240) -> np.ndarray:
    y = -np.log((pxi.astype(float) + 1) / Io).T

    # determine concentrations of the individual stains
    return np.linalg.lstsq(stain_matrix, y, rcond=None)[0]


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
