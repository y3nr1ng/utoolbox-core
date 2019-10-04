import logging
from math import ceil
import os

import cupy as cp

from utoolbox.parallel import RawKernelFile

__all__ = ["ImageResize", "imresize"]

logger = logging.getLogger(__name__)


class ImageResize(object):
    def __init__(self, shape, interp="bilinear", tile_sz=16):
        cu_file = os.path.join(os.path.dirname(__file__), "imresize.cu")
        self._kernels = RawKernelFile(cu_file)
        self._tile_sz = tile_sz

        self._shape = shape

        # pre-calculate grid size
        ny, nx = shape
        self._grid_sz = (int(ceil(nx / tile_sz)), int(ceil(ny / tile_sz)))

    def __call__(self, x):
        """
        Args:
            x (cp.ndarray): Input data.
            in_place (bool, optional): Write result into provided array.
        """
        nv, nu = self.shape
        ny, nx = x.shape

        out_buf = cp.empty(self.shape)
        self._kernels["imresize_bilinear_kernel"](
            self.grid_sz, (self._tile_width,) * 2, (out_buf, nu, nv, x, nx, ny)
        )
        return out_buf

    @property
    def grid_sz(self):
        return self._grid_sz

    @property
    def shape(self):
        return self._shape

    @property
    def tile_sz(self):
        return self._tile_sz


def imresize(arr, shape, interp="bilinear"):
    """
    Resize an image.

    Args:
        arr (cp.ndarray): The array of image to be resized.
        shape (tuple): Size of the output image.
        interp (str, optional): Interpolation to use for re-sizing ('nearest', 
            or 'bilinear').

    Returns:
        imresize (cp.ndarray): The resized array of image.
    """
    with ImageResize(shape, interp) as func:
        return func(arr)

