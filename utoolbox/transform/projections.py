from enum import auto, Flag
import logging
from math import ceil
import os

import cupy as cp
import numpy as np

from utoolbox.util.decorator import run_once

__all__ = ["Orthogonal"]

logger = logging.getLogger(__name__)

###
# region: Kernel definitions
###

cu_file = os.path.join(os.path.dirname(__file__), "projections.cu")

with open(cu_file, "r") as fd:
    source = fd.read()
z_proj_kernel = cp.RawKernel(source, "z_proj_kernel")
y_proj_kernel = cp.RawKernel(source, "y_proj_kernel")
x_proj_kernel = cp.RawKernel(source, "x_proj_kernel")

###
# endregion
###


class Orthogonal(object):
    """Orthogonal maximum intensity projections."""

    def __init__(self, volume, block_sz=(16, 16)):
        """
        :param np.ndarray volume: volume to project
        :param tuple(int,int) block_sz: kernel block size
        """
        self._volume = volume
        self._block_sz = block_sz

    def __enter__(self):
        self._volume = cp.asarray(self._volume)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._volume = None

    @property
    def xy(self):
        return self._z_proj()

    @property
    def xz(self):
        return self._y_proj()

    @property
    def yz(self):
        return self._x_proj()

    @run_once
    def _z_proj(self):
        nz, ny, nx = self._volume.shape
        view = cp.empty((ny, nx), self._volume.dtype)

        nbx, nby = self._block_sz
        grid_sz = (int(ceil(nx / nbx)), int(ceil(ny / nby)))
        z_proj_kernel(grid_sz, self._block_sz, (view, self._volume, nx, ny, nz))

        return cp.asnumpy(view)

    @run_once
    def _y_proj(self):
        nz, ny, nx = self._volume.shape
        view = cp.empty((nz, nx), self._volume.dtype)

        nbx, nbz = self._block_sz
        grid_sz = (int(ceil(nx / nbx)), int(ceil(nz / nbz)))
        y_proj_kernel(grid_sz, self._block_sz, (view, self._volume, nx, ny, nz))

        return cp.asnumpy(view)

    @run_once
    def _x_proj(self):
        nz, ny, nx = self._volume.shape
        view = cp.empty((nz, ny), self._volume.dtype)

        nby, nbz = self._block_sz
        grid_sz = (int(ceil(ny / nby)), int(ceil(nz / nbz)))
        x_proj_kernel(grid_sz, self._block_sz, (view, self._volume, nx, ny, nz))

        return cp.asnumpy(view)
