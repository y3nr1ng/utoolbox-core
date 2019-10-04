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

__all__ = ["PeronaMalik2D", "PeronaMalik3D"]

logger = logging.getLogger(__name__)


class PeronaMalik2D(object):
    """
    Perona-Malik anisotropic smoothing filter in 2D.

    Args:
        threshold (float, optional): Conduction function threshold.
        niter (float, optiona): Number of iterations.
        tile_width (int, optional): Tile size during kernel launch.
    """

    def __init__(self, threshold=30.0, niter=16, tile_width=16):
        cu_file = os.path.join(os.path.dirname(__file__), "perona_malik.cu")
        self._kernels = RawKernelFile(cu_file, tile_width=tile_width)
        self._tile_width = tile_width

        self._threshold, self._niter = np.float32(threshold), niter

    def __call__(self, x, in_place=True):
        """
        Args:
            x (cp.ndarray): Input data.
            in_place (bool, optional): Write result into provided array.
        """
        ny, nx = x.shape
        grid_sz = (int(ceil(nx / self._tile_width)), int(ceil(ny / self._tile_width)))

        in_buf = x if in_place else cp.copy(x)
        out_buf = cp.empty_like(in_buf)
        for _ in range(self._niter):
            self._kernels["perona_malik_2d_kernel"](
                grid_sz,
                (self._tile_width,) * 2,
                (out_buf, in_buf, np.float32(self._threshold), nx, ny),
            )
            in_buf, out_buf = out_buf, in_buf
        return in_buf


class PeronaMalik3D(object):
    """
    Perona-Malik anisotropic smoothing filter in 3D.

    Args:
        threshold (float, optional): Conduction function threshold.
        niter (float, optiona): Number of iterations.
        tile_width (int, optional): Tile size during kernel launch.
    """

    def __init__(self, threshold=30.0, niter=16, tile_width=8):
        cu_file = os.path.join(os.path.dirname(__file__), "perona_malik.cu")
        self._kernels = RawKernelFile(cu_file, tile_width=tile_width)
        self._tile_width = tile_width

        self._threshold, self._niter = np.float32(threshold), niter

    def __call__(self, x, in_place=True):
        """
        Args:
            x (cp.ndarray): Input data.
            in_place (bool, optional): Write result into provided array.
        """
        nz, ny, nx = x.shape
        grid_sz = (
            int(ceil(nx / self._tile_width)),
            int(ceil(ny / self._tile_width)),
            int(ceil(nz / self._tile_width)),
        )

        in_buf = x if in_place else cp.copy(x)
        out_buf = cp.empty_like(in_buf)
        for _ in range(self._niter):
            self._kernels["perona_malik_3d_kernel"](
                grid_sz,
                (self._tile_width,) * 3,
                (out_buf, in_buf, np.float32(self._threshold), nx, ny, nz),
            )
            in_buf, out_buf = out_buf, in_buf
        return in_buf


if __name__ == "__main__":

    from imageio import volread, volwrite
    from utoolbox.exposure.rescale_intensity import RescaleIntensity

    in_data = volread("mito.tif")

    _, _, nc = in_data.shape
    pm = PeronaMalik3D(threshold=10, niter=16)

    in_data = in_data.astype(np.float32)
    in_data = cp.asarray(in_data)

    out_data = pm(in_data)

    ri = RescaleIntensity()
    out_data = ri(out_data, out_range=np.uint16)

    out_data = cp.asnumpy(out_data)
    out_data = out_data.astype(np.uint16)

    volwrite("result.tif", out_data)
