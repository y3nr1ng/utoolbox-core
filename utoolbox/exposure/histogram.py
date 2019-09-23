import logging
from math import ceil
import os

import cupy as cp
import numpy as np

from utoolbox.utils.decorator import run_once

__all__ = ["Histogram", "histogram"]

logger = logging.getLogger(__name__)

###
# region: Kernel definitions
###

cu_file = os.path.join(os.path.dirname(__file__), "histogram.cu")

with open(cu_file, "r") as fd:
    source = fd.read()
hist_atom_kernel = cp.RawKernel(source, "hist_atom_kernel")
hist_accu_kernel = cp.RawKernel(source, "hist_accu_kernel")

###
# endregion
###


class Histogram(object):
    def __init__(self, data, n_bins=256, block_sz=(16, 16)):
        self._partial = None

        if data.dtype != np.uint16:
            raise TypeError("only support uint16")
        self._data = data
        self._n_bins = n_bins

        self._block_sz = block_sz

        # pre-compute grid size
        _, ny, nx = data.shape
        nbx, nby = block_sz
        self._grid_sz = (int(ceil(nx / nbx)), int(ceil(ny / nby)))

    def __enter__(self):
        # holding area for
        ngx, ngy = self._grid_sz
        self._partial = cp.empty((ngx * ngy, self._n_bins), cp.uint32)

        # upload to device
        self._data = cp.asarray(self._data)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._partial = None
        self._data = None

    @property
    def histogram(self):
        return self._histogram()

    @run_once
    def _histogram(self):
        nz, ny, nx = self._data.shape
        ngx, ngy = self._grid_sz

        # size of the shared memory
        nbytes = np.dtype(np.uint32).itemsize * self._n_bins

        # TODO force scaling for [0, 65536) range
        scale = 2 ** 16 / self._n_bins
        hist_atom_kernel(
            self._grid_sz,
            self._block_sz,
            args=(
                self._partial,
                self._data,
                self._n_bins,
                np.float32(scale),
                nx,
                ny,
                nz,
            ),
            shared_mem=nbytes,
        )

        hist = cp.empty((self._n_bins,), cp.uint32)
        nthreads = 1024 if self._n_bins > 1024 else self._n_bins
        hist_accu_kernel(
            (int(ceil(self._partial.size / nthreads)),),
            (nthreads,),
            (hist, self._partial, self._n_bins, ngx * ngy),
        )

        return cp.asnumpy(hist)


def histogram(data, n_bins=256):
    """Helper function that wraps :class:`.Histogram` for one-off use."""
    try:
        data = np.squeeze(data, axis=0)
    except ValueError:
        # unable to squeeze
        pass

    if data.ndim == 2:
        hist, _ = np.histogram(data, n_bins, (0, 2 ** 16))
        return hist
    elif data.ndim == 3:
        with Histogram(data, n_bins) as h:
            return h.histogram
    else:
        raise ValueError("unsupported dimension")
