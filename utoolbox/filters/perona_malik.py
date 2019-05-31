"""
Perona-Malik anisotropic smoothing filter [Sergeevich]_.

.. [Sergeevich] Employing the Perona - Malik anisotropic filter for the problem of landing site detection: https://github.com/Galarius/pm-ocl
"""
import logging
from math import ceil
import os

import cupy as cp
import numpy as np

from utoolbox.parallel import RawKernelFile

logger = logging.getLogger(__name__)


class PeronaMalik2D(object):
    def __init__(self, shape, dtype=None, tile_width=16, threshold=30., lamb=0.25):
        cu_file = os.path.join(os.path.dirname(__file__), "perona_malik.cu")
        self._kernels = RawKernelFile(cu_file, tile_width=tile_width)
        self._tile_width = tile_width

        self._threshold, self._lamb = threshold, lamb

        # TODO wrap this into class, automate the determine process
        self._buffer, self._shape, self._dtype = None, shape, dtype

    def __enter__(self):
        self._buffer = cp.empty(self._shape, cp.float32)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._buffer = None

    def __call__(self, x):
        """
        void (
            float *dst,
            const float *src,
            const float thre, const float lambda,
            const int nx, const int ny
        )
        """
        ny, nx = self._buffer.shape
        grid_sz = (int(ceil(nx / self._tile_width)), int(ceil(ny / self._tile_width)))
        self._kernels["perona_malik_2d_kernel"](
            grid_sz,
            (self._tile_width,) * 2,
            (self._buffer, cp.asarray(x), self.threshold, self.lamb, nx, ny),
        )
        return cp.asnumpy(self._buffer)

    @property
    def threshold(self):
        return self._threshold

    @property
    def lamb(self):
        return self._lamb


def perona_malik():
    """Helper function that wraps :class:`.PeronaMalk` for one-off use."""
    pass
