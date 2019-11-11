import logging
from math import ceil
import os

import cupy as cp

from utoolbox.parallel import RawKernelFile

__all__ = ["median"]

logger = logging.getLogger(__name__)


def median(image, kernel_size=3, tile_width=8):
    cu_file = os.path.join(os.path.dirname(__file__), "median.cu")
    kernels = RawKernelFile(
        cu_file, kernel_radius=kernel_size // 2, tile_width=tile_width
    )
    kernel = kernels[f"median_{image.ndim}d_kernel"]

    nz, ny, nx = image.shape
    grid_sz = (
        int(ceil(nx / tile_width)),
        int(ceil(ny / tile_width)),
        int(ceil(nz / tile_width)),
    )

    image = cp.asarray(image)
    result = cp.empty_like(image)

    kernel(grid_sz, (tile_width,) * 3, (result, image, nx, ny, nz))

    return result
