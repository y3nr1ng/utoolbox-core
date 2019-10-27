import logging

import numpy as np
from scipy.ndimage import zoom

__all__ = ["BaSiC"]

logger = logging.getLogger(__name__)

##


def dct(a, ww, ind):
    """1D discrete cosine transform

    DCT operates along the first dimension. 

    Args:
        a (ndarray): array to perform DCT
        ww (ndarray): weights
        ind (ndarray): rearrangement array for a
    """
    return ww - np.fft.fft(a[ind], axis=0).real


def mirt_dct2(a):
    """
    2-D discrete cosine transform
    """
    ww, ind = [], []
    for n in a.shape:
        _ww = np.arange(0, n)
        _ww = 2 * np.exp(((-1j * np.pi) / (2 * n)) * _ww) / np.sqrt(2 * n)
        _ww[0] /= np.sqrt(2)
        ww.append(_ww)

        _ind = np.concatenate(
            (np.arange(0, n, 2), np.flipud(np.arange(1, n, 2))), axis=0
        )
        _ind += np.arange(0, a.size - n + 1, n)  # TODO validate this from MATLAB
        ind.append(_ind)

    return dct(dct(a, ww[0], ind[0]).T, ww[1], ind[1]).T


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
        self._lambda_flat, self._lambda_dark = lambda_flat, lambda_dark
        self._tol = tol
        self._max_iter = max_iter  # pca

        self._wsize = wsize
        self._max_reweight_iter = max_reweight_iter
        self._epsilon = epsilon
        self._reweight_tol = reweight_tol

    def __call__(self, x):
        x = x.astype(np.float32)

        # reshape to working size
        nz, ny, nx = x.shape
        D = zoom(x, (nz, self._wsize / ny, self._wsize / nx), order=1)

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
        logger.debug(
            f"lambda_flat: {self._lambda_flat}, lambda_dark: {self._lambda_dark}"
        )

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
        for i in range(self._max_reweight_iter):
            logger.debug(f"reweighting iteration {i}")

            XA, XE, XAoffset = self._inexact_alm_rspca_l1()
            XE_norm = XE / (XA.mean() + self._tol)

            weight = 1 / (np.abs(XE_norm) + self._epsilon)
            weight /= weight.mean()

            temp = XA.mean(axis=0) - XAoffset
            flatfield_curr = temp / temp.mean()
            darkfield_curr = XAoffset

            mad_flatfield = np.abs(flatfield_curr - flatfield_last).sum()
            mad_flatfield /= np.abs(flatfield_last).sum()
            mad_darkfield = np.abs(darkfield_curr - darkfield_last).sum()
            if mad_darkfield < 1e-7:  # machine epsilon
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
        if self._darkfield:
            # summarize darkfield
            darkfield = zoom(XAoffset, zoom_factor, order=1)
            return flatfield, darkfield
        else:
            return flatfield

    def _inexact_alm_rspca_l1(self, D, weight=1):
        """
        Inexact augmented Lagrange multiplier method for sparse low-rank matrix recovery.

            while ~converged
                minimize (inexactly, update A and E only once)
                L(W, E, Y, u) = |E|_1 + lambda * |W|_1 + <Y2, D-repmat(QWQ^T)-E> + +mu/2 * |D-repmat(QWQ^T)-E|_F^2
                Y1 = Y1 + \mu * (D-repmat(QWQ^T)-E)
                \mu = \rho * \mu
            end

        Args:
            D (ndarray): N x M x M matrix, N observations of M x M matrices

        References:
            - Adapted from `BaSiC` (QSCD), https://github.com/QSCD/BaSiC
            - Modified from `Robust PCA`
        """

        pass


##


def basic():
    pass


##

if __name__ == "__main__":
    import coloredlogs
    import imageio
    import numpy as np

    from utoolbox.data.datastore import ImageFolderDatastore

    logging.getLogger("tifffile").setLevel(logging.ERROR)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    ds = ImageFolderDatastore("basic_wsi", read_func=imageio.imread, create_new=False)
    # collect images
    data = list(ds.values())
    data = np.stack(data)
    logger.info(f"data.shape={data.shape}, data.dtype={data.dtype}")

    algo = BaSiC(darkfield=True)
    flatfield, darkfield = algo(data)

    # save
    imageio.imwrite("flatfield.tif", flatfield.astype(np.float32))
    imageio.imwrite("darkfield.tif", darkfield.astype(np.float32))
