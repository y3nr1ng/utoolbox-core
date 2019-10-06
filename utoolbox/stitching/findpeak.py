import logging
from math import ceil
import os

import cupy as cp
import numpy as np

from utoolbox.parallel import RawKernelFile
from utoolbox.transform.projections import Orthogonal

__all__ = ["PeronaMalik2D", "PeronaMalik3D"]

logger = logging.getLogger(__name__)


class FindPeak3D(object):
    def __init__(self, threshold=None, tile_width=8):
        cu_file = os.path.join(os.path.dirname(__file__), "findpeak.cu")
        self._kernels = RawKernelFile(cu_file, tile_width=tile_width)
        self._tile_width = tile_width

        self._threshold = threshold

    def __call__(self, x):
        """
        Args:
            x (cp.ndarray): input data
        """
        nz, ny, nx = x.shape
        grid_sz = (
            int(ceil(nx / self._tile_width)),
            int(ceil(ny / self._tile_width)),
            int(ceil(nz / self._tile_width)),
        )

        buf = cp.copy(x)
        if not isinstance(x.dtype, cp.uint16):
            logger.info(".. rescaling to uint16")
            buf = buf / buf.max() * 65535
            buf = buf.astype(cp.uint16)

        threshold = np.uint16(
            self._threshold if self._threshold else FindPeak3D.find_threshold(buf)
        )
        logger.debug(f".. threshold: {threshold}")
        self._kernels["find_peak_3d_kernel"](
            grid_sz,
            (self._tile_width,) * 3,
            (buf, buf, threshold, nx, ny, nz)
        )

        coords = cp.nonzero(buf)
        coords = [cp.asnumpy(_coords) for _coords in coords]
        logger.debug(f'found {len(coords[0])} candidates')
        peak_list = []
        for coord in zip(*coords):
            print(coord)
            peak_list.append(coord)
        peak_list.sort(key=lambda x: x[0], reverse=True)

        return peak_list

    ##

    @property
    def threshold(self):
        return self._threshold

    ##

    @staticmethod
    def find_threshold(x):
        """
        Find maximum of minimum of maximum along each dimension.

        Args:
            x (cp.ndarray): input data
        """
        ortho = Orthogonal(x, "max")
        return max(
            cp.asnumpy(ortho.xy.min()),
            cp.asnumpy(ortho.xz.min()),
            cp.asnumpy(ortho.yz.min()),
        )

