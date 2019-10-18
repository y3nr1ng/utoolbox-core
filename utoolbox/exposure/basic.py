import logging

import numpy as np
from scipy.ndimage import zoom

from utoolbox.transform import imresize

__all__ = ["BaSiC"]

logger = logging.getLogger(__name__)

##


def dct(array, weights):
    pass


def mirt_dct2(data):
    """
    2-D discrete cosine transform
    """
    pass


##


class BaSiC(object):
    """
    Args:
        darkfield (bool, optional): estimate darkfield
            Set to True if input images are fluorescence image with strong darkfield 
            contribution.
        lambda_flat (float, optional): regularization strength for flatfield
        lambda_dark (float, optional): regularization strength for darkfield
        tol (float, optional): tolerance of error during optimization
        max_iter (int, optional): number of iterations
    """

    def __init__(
        self,
        darkfield=False,
        lambda_flat=None,
        lambda_dark=None,
        tol=1e-6,
        max_iter=500,
        # advance options
        estimation_mode="L0",
        wsize=128,
        max_reweight_iter=10,
        epsilon=0.1,
        varying_coeff=True,
        reweight_tol=1e-3,
    ):
        self._darkfield = darkfield
        self._tol = tol
        self._max_iter = max_iter

        self._wsize = wsize
        self._epsilon = epsilon
        self._reweight_tol = reweight_tol

    def __call__(self, x):
        x = x.astype(np.float32)

        # reshape to working size
        ny, nx = x.shape[1:]
        D = zoom(x, (self._wsize / ny, self._wsize / nx), order=1)

        # calculate weights
        meanD = D.mean(axis=0)
        meanD /= meanD.mean()
        W_meanD = mirt_dct2(meanD)

        # calculate lambda if needed
        if (self._lambda_flat is None) or (self._lambda_dark is None):
            stats = np.abs(W_meanD).sum() / 400
            if self._lambda_flat is None:
                self._lambda_flat = stats * 0.5
            if self._lambda_dark is None:
                self._lambda_dark = stats * 0.2

        D = np.sort(D, axis=0)
        XAoffset = np.zeros((self._wsize, self._wsize), dtype=np.float32)

        # reweighting iterations
        weight = np.ones_like(D)
        flatfield_last, flatfield_curr = (
            np.ones((self._wsize, self._wsize), dtype=np.float32),
            None,
        )
        darkfield_last, darkfield_curr = (
            np.random.randn(self._wsize, self._wsize).astype(np.float32),
            None,
        )
        XA, XAoffset = None, None
        for i in range(self._max_iter):
            logger.debug(f"reweighting iteration {i}")

            XA, XE, XAoffset = inexact_alm_rspca_l1()
            XE_norm = XE / (XA.mean() + self._tol)

            weight = 1 / (np.abs(XE_norm) + self._epsilon)
            weight /= weight.mean()

            temp = XA.mean(axis=0) - XAoffset
            flatfield_curr = temp / temp.mean()
            darkfield_curr = XAoffset

            mad_flatfield = np.abs(flatfield_curr - flatfield_last).sum()
            mad_flatfield /= np.abs(flatfield_last).sum()
            mad_darkfield = np.abs(darkfield_curr - darkfield_last).sum()
            if mad_darkfield < 1e-7:
                mad_darkfield = 0
            else:
                mad_darkfield /= max(np.abs(darkfield_last).sum(), self._tol)

            flatfield_last, darkfield_last = flatfield_curr, darkfield_curr

            if max(mad_flatfield, mad_darkfield) < self._reweight_tol:
                break
        else:
            logger.warning("maximum reweighting iterations reached but not converged")

        zoom_factor = (ny / self._wsize, nx / self._wsize)
        # summarize flatfield
        shading = XA.mean(axis=0) - XAoffset
        flatfield = zoom(shading, zoom_factor, order=1)
        flatfield /= flatfield.mean()
        # summarize darkfield
        darkfield = zoom(XAoffset, zoom_factor, order=1)

        return flatfield, darkfield


##


def basic():
    pass
