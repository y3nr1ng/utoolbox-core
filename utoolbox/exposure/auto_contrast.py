import logging

import cupy as cp
import numpy as np

__all__ = ["AutoContrast"]
logger = logging.getLogger(__name__)


class AutoContrast(object):
    def __init__(self, n_bins=256, auto_threshold=5000, block_sz=(512,)):
        self._n_bins = n_bins
        self._auto_threshold = auto_threshold

    def __call__(self, data):
        # histogram
        hist, edges = cp.histogram(data, bins=self.n_bins)
        # recenter
        hist, edges = cp.asnumpy(hist), cp.asnumpy(edges)
        edges = ((edges[:-1] + edges[1:]) / 2).astype(data.dtype)

        nelem = data.size
        limit, threshold = nelem / 10, nelem / self.auto_threshold

        # minimum
        hmin = -1
        for i, cnt in enumerate(hist):
            if cnt > limit:
                continue
            if cnt > threshold:
                hmin = i
                break

        # maximum
        hmax = -1
        for i, cnt in reversed(list(enumerate(hist))):
            if cnt > limit:
                continue
            if cnt > threshold:
                hmax = i
                break

        vmin, vmax = edges[hmin], edges[hmax]
        logger.info(f"auto adjust contrast to [{vmin}, {vmax}]")

        #TODO lookup kernel
    @property
    def n_bins(self):
        return self._n_bins

    @property
    def auto_threshold(self):
        return self._auto_threshold
