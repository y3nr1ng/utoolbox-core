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
    def __init__(self, shape, dtype=None, tile_width=16, threshold=30.0, lamb=0.25):
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

    def __call__(self, x, niter=16):
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

        in_buf, out_buf = cp.asarray(x), self._buffer
        for _ in range(niter):
            self._kernels["perona_malik_2d_kernel"](
                grid_sz,
                (self._tile_width,) * 2,
                (
                    out_buf,
                    in_buf,
                    np.float32(self.threshold),
                    np.float32(self.lamb),
                    nx,
                    ny,
                ),
            )
            in_buf, out_buf = out_buf, in_buf
        return cp.asnumpy(in_buf)

    @property
    def threshold(self):
        return self._threshold

    @property
    def lamb(self):
        return self._lamb


def perona_malik():
    """Helper function that wraps :class:`.PeronaMalk` for one-off use."""
    pass


if __name__ == "__main__":
    """
    from imageio import imread, imwrite

    in_data = imread("heli_in.png")

    ny, nx, nc = in_data.shape
    with PeronaMalik2D((ny, nx), in_data.dtype, threshold=10, lamb=0.25) as pm:
        out_data = []
        for ic in range(nc):
            in_data_ = in_data[..., ic]

            print(in_data_)

            in_data_ = in_data_.astype(np.float32)
            #in_data_ /= in_data_.max()
            
            for _ in range(16):
                out_data_ = pm(in_data_)
                out_data_, in_data_ = in_data_, out_data_
            
            #out_data_ /= out_data_.max()
            #out_data_ *= 255

            out_data_ = out_data_.astype(np.uint8)
            out_data.append(out_data_)

            print(out_data_)

            print()

    out_data = np.stack(out_data, axis=-1)
    imwrite("heli_out.png", out_data)
    """

    from imageio import volread, volwrite

    in_data = volread("cell_in.tif")

    nz, ny, nx = in_data.shape
    with PeronaMalik2D((ny, nx), in_data.dtype, threshold=100, lamb=0.25) as pm:
        out_data = []
        for iz in range(nz):
            print(iz)

            in_data_ = in_data[iz, ...]

            in_data_ = in_data_.astype(np.float32)
            # in_data_ /= in_data_.max()

            out_data_ = pm(in_data_, niter=32)

            # out_data_ /= out_data_.max()
            # out_data_ *= 255

            out_data_ = out_data_.astype(np.uint16)
            out_data.append(out_data_)

    out_data = np.stack(out_data, axis=0)
    volwrite("cell_out.tif", out_data)
