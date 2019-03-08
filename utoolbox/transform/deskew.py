import logging
from math import degrees, radians

import cupy
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'Deskew', 
    'deskew'
]

class Deskew(object):
    """Restore the actual spatial size of acquired lightsheet data."""
    def __init__(self, angle=32.8, dr=.108, dz=0.5, rotate=True, 
                 resample=False):
        """
        :param float angle: target objective angle
        :param float dr: lateral resolution
        :param float dz: step size of the piezo stage
        :param bool rotate: rotate the result to conventional axis
        :param bool resample: resample the data to isotropic voxel size
        """
        self._angle = radians(angle)
        self._dr, self._dz = dr, dz
        self._resample = resample
        self._rotate = rotate

        self._in_buffer, self._out_buffer = None, None
    
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def angle(self):
        """
        Target objective angle in degrees
        """
        return degrees(self._angle)
    
    @property
    def dr(self):
        """Lateral resolution in microns."""
        return self._dr

    @property
    def dz(self):
        """Axial resolution in microns."""
        return self._dz
    
    @property
    def resample(self):
        """Resample the data to isotropic voxel size."""
        return self._resample
    
    @property
    def rotate(self):
        """Rotate the result to conventional axis."""
        return self._rotate
    
    def _estimate_resolution(self):
        """
        Estimate output lateral and axial resolution.
        
        :return: tuple of (lateral, axial) resolution
        :rtype: tuple(float,float)
        """
        pass
    
    def _estimate_shape(self):
        """
        Estimate output shape.
        
        :return: tuple of shpae in numpy format (nz, ny, nx)
        :rtype: tuple(int,int,int)
        """
        pass

def deskew(data, **kwargs):
    pass