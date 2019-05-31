"""
Perona-Malik anisotropic smoothing filter [Sergeevich]_.

.. [Sergeevich] Employing the Perona - Malik anisotropic filter for the problem of landing site detection: https://github.com/Galarius/pm-ocl
"""
import logging
import os

import cupy as cp
import numpy as np

from utoolbox.container import RawKernelFile

logger = logging.getLogger(__name__)


class PeronaMalik2D(object):
    def __init__(self, volume):
        cu_file = os.path.join(os.path.dirname(__file__), "perona_malik.cu")
        self._kernels = RawKernelFile(cu_file, tile_width=16)[]

        # perona_malik_2d_kernel = self._kernels["perona_malik_2d_kernel"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run_once(self):
        pass

    def run(self):
        pass


def perona_malik():
    """Helper function that wraps :class:`.PeronaMalk` for one-off use."""
    pass
