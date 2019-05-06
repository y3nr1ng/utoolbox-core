from functools import reduce
import logging
from math import ceil, cos, degrees, radians, sin
from operator import mul
import os

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'Deskew', 
    'deskew'
]

###
# region: kernel definitions
###

cu_file = os.path.join(os.path.dirname(__file__), 'deskew.cu')

ushort_to_float = cp.ElementwiseKernel(
    'uint16 src', 'float32 dst',
    'dst = (float)src',
    'ushort_to_float'
)

float_to_ushort = cp.ElementwiseKernel(
    'float32 src', 'uint16 dst',
    'dst = (unsigned short)src',
    'float_to_ushort'
)

###
# endregion
###

class Deskew(object):
    """Restore the actual spatial size of acquired lightsheet data."""
    def __init__(self, angle=32.8, dr=.108, dz=0.5, rotate=True):
        """
        :param float angle: target objective angle
        :param float dr: lateral resolution
        :param float dz: step size of the piezo stage
        :param bool rotate: rotate the result to conventional axis
        """
        self._angle = radians(angle)
        self._in_res = (dr, dz)
        self._rotate = rotate

        #DEBUG
        if rotate:
            raise NotImplementedError("not yet supported")

        self._out_res = self._estimate_resolution()

        self._in_shape, self._out_shape = None, None
        self._template = None

    @property
    def angle(self):
        """
        Target objective angle in degrees
        """
        return degrees(self._angle)

    @property
    def dr(self):
        """Output lateral resolution in microns."""
        return self._out_res[0]
    
    @property
    def dz(self):
        """Output axial resolution in microns."""
        return self._out_res[1]
    
    @property
    def rotate(self):
        """Rotate the result to conventional axis."""
        return self._rotate

    @property
    def shape(self):
        """Final shape after deskew."""
        return self._out_shape
    
    def _estimate_resolution(self):
        """
        Estimate output lateral and axial resolution.
        
        :return: (lateral, axial) resolution
        :rtype: tuple(float,float)
        """
        dr, dz0 = self._in_res
        dz = dz0 * sin(self._angle)
        return (dr, dz)

    def _refresh_internal_buffers(self, ref):
        logger.info("updating internal buffers")

        # reference for interpolation, on device
        self._template = cp.empty_like(ref, dtype=cp.float32)

        # store shape info
        self._in_shape = ref.shape
        self._out_shape = self._estimate_shape(ref)

    def _estimate_shape(self, ref):
        """
        Estimate output shape.
        
        :return: numpy shape format, (nz, ny, nx)
        :rtype: tuple(int,int,int)
        """
        #DEBUG
        return ref.shape

        dx = self._estimate_shift()
        nz, ny, nx0 = ref.shape
        nx = ceil(nx0 + dx * (nz-1))
        return (nz, ny, nx)

    def _estimate_shift(self):
        """
        Estimate pixel shift per layer.
        
        :return: pixel shift
        :rtype: float
        """
        dr0, dz0 = self._in_res
        return dz0 * cos(self._angle) / dr0

    def _upload(self, src):
        """
        Uplaod data to device.

        :param np.ndarray src: source
        """
        ushort_to_float(cp.asarray(src), self._template)

    def _download(self, dst):
        """
        Download data from device.

        :param np.ndarray dst: destination

        .. note::
            Pinned memory and device memory are reused throughout the process.
        """
        buffer = cp.asarray(dst)
        float_to_ushort(self._template, buffer)
        dst[...] = cp.asnumpy(buffer)

    def run(self, data, out=None):
        try:
            if data.shape != self._in_shape:
                raise RuntimeError("incompatible buffer")
        except:
            self._refresh_internal_buffers(data)

        self._upload(data)

        #TODO execute conversion

        if out is None:
            out = np.empty(self._out_shape, data.dtype)
        self._download(out)
        return out
    
def deskew(data, **kwargs):
    """Helper function that wraps :class:`.Deskew` for one-off use."""
    pass