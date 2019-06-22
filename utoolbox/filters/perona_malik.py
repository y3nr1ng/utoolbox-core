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
    """
    Perona-Malik anisotropic smoothing filter in 2D.

    Args:
        threshold (float, optional): Conduction function threshold.
        lamb (float, optional): Gradient coefficient.
        niter (float, optiona): Number of iterations.
        tile_width (int, optional): Tile size during kernel launch.
    """

    def __init__(self, threshold=30.0, lamb=0.25, niter=16, tile_width=16):
        cu_file = os.path.join(os.path.dirname(__file__), "perona_malik.cu")
        self._kernels = RawKernelFile(cu_file, tile_width=tile_width)
        self._tile_width = tile_width

        self._threshold, self._lamb, self._niter = (
            np.float32(threshold),
            np.float32(lamb),
            niter,
        )

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
                (
                    out_buf,
                    in_buf,
                    np.float32(self._threshold),
                    np.float32(self._lamb),
                    nx,
                    ny,
                ),
            )
            in_buf, out_buf = out_buf, in_buf
        return in_buf


if __name__ == "__main__":

    from imageio import imread, imwrite

    in_data = imread("heli_in.png")

    _, _, nc = in_data.shape
    pm = PeronaMalik2D(threshold=10, lamb=0.25, niter=1)

    out_data = []
    for ic in range(nc):
        in_data_ = in_data[..., ic]

        in_data_ = in_data_.astype(np.float32)
        in_data_ = cp.asarray(in_data_)
        # in_data_ /= in_data_.max()

        out_data_ = pm(in_data_)

        # out_data_ /= out_data_.max()
        # out_data_ *= 255

        out_data_ = cp.asnumpy(out_data_)
        out_data_ = out_data_.astype(np.uint8)
        out_data.append(out_data_)

        print()

    out_data = np.stack(out_data, axis=-1)
    imwrite("heli_out.png", out_data)
