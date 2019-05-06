# pylint: disable=E1101
import logging
import numpy as np
import pycuda.driver as cuda
from pycuda.elementwise import ElementwiseKernel

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

__all__ = [
    'RescaleIntensity'
]

logger = logging.getLogger(__name__)

class RescaleIntensity(metaclass=AbstractAlgorithm):
    def __call__(self, image, in_range=None, out_range=None):
        dtype = image.dtype
        image = self._upload(image)

        imin, imax = self._intensity_range(image, in_range)
        if out_range is None:
            out_range = dtype
        omin, omax = self._intensity_range(image, out_range, clip=(imin>=0))
        logger.debug("in={}, out={}".format((imin, imax), (omin, omax)))

        image = self._scale_by_range(image, (imin, imax), (omin, omax), dtype)
        image = self._download(image)

        return image.astype(dtype)

    @interface
    def _upload(self, image):
        """Upload data to processing unit."""
        pass

    def _intensity_range(self, image, range_values=None, clip=False):
        if range_values is None:
            imin, imax = image.min(), image.max()
        elif isinstance(range_values, tuple):
            imin, imax = range_values
        elif np.issubdtype(range_values, np.integer):
            imin, imax = np.iinfo(range_values).min, np.iinfo(range_values).max
        elif np.issubdtype(range_values, np.inexact):
            imin, imax = -1., 1.
        else:
            raise TypeError("unknown range definition")
        
        if clip:
            imin = 0

        return imin, imax
    
    @interface
    def _scale_by_range(self, image, in_range, out_range, dtype):
        pass

    @interface
    def _download(self, image):
        return image

class RescaleIntensity_CPU(RescaleIntensity):
    _strategy = ImplTypes.CPU_ONLY

    def _upload(self, image):
        return image

    def _scale_by_range(self, image, in_range, out_range, dtype):
        imin, imax = in_range
        omin, omax = out_range

        image = np.clip(image, imin, imax)

        image = (image - imin) / float(imax - imin)
        return np.array(image * (omax - omin) + omin, dtype=dtype)

    def _download(self, image):
        return image

class RescaleIntensity_GPU(RescaleIntensity):
    _strategy = ImplTypes.GPU

    def __init__(self):
        self.image_h = None

    def _upload(self, image_h):
        self.image_h = image_h
        image_g = cuda.pagelocked_empty_like(
            image_h, mem_flags=cuda.host_alloc_flags.DEVICEMAP
        )
        image_g[:] = image_h
        return image_g
    
    def _scale_by_range(self, image, in_range, out_range, dtype):
        imin, imax = in_range
        omin, omax = out_range

        rescale = ElementwiseKernel(
            "float *dst, float *src, float imin, float imax, float omin, float omax",
            "dst[i] = (((src[i] < imin) ? imin : (src[i] > imax) ? imax : src[i])-imin)/(imax-imin)"
        )
        rescale(image, image, imin, imax, omin, omax)

        return image

    def _download(self, image_g):
        image_h = self.image_h[:]
        image_h[:] = image_g
        self.image_h = None
        return image_h