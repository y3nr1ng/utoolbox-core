"""
Modified pyramids routines from skimage for anisotropic 3D volumes.
"""
import logging
from math import ceil

import numpy as np
import scipy.ndimage as ndi

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

__all__ = [
    'GaussianPyramid'
]

logger = logging.getLogger(__name__)

def _smooth(image, res, sigma, mode):
    """ 
    Return data smoothed by the Gaussian filter.

    Note
    ----
    To avoid further quality degrade, sigma is normalized by the axis with 
    finest resolution.
    """
    assert len(res) == len(sigma)
    res, sigma = list(res), list(sigma)

    mres = min(res)
    res = [x/mres for x in res]
    # rescale sigma
    sigma = [s/r for r, s in zip([res, sigma])]
    logger.debug("new sigma {}".format(sigma))

    smoothed = np.empty_like(image)
    ndi.gaussian_filter(image, sigma, output=smoothed, mode=mode)
    return smoothed

def pyramid_reduce(image, res, downscale=2, sigma=None, mode='reflect'):
    """
    Smooth and then downsample image.
    """
    image = img_as_float32(image)
    
    downscale = float(downscale)
    out_shape = tuple([ceil(s/downscale) for s in image.shape])
    if sigma is None:
        sigma = [2*downscale/6.] * len(out_shape)
    
    smoothed = _smooth(image, res, sigma, mode)
    out = resize(smoothed, out_shape, order=1, mode=mode, anti_aliasing=False)
    return out

def pyramid_expand():
    """
    Upsample and then smooth image.
    """
    pass

class GaussianPyramid(metaclass=AbstractAlgorithm):
    def __init__(self, size, res, downscale=2, sigma=None, order=1):
        pass    

    @interface
    def __call__(self, data, level=None):
        pass

    @interface
    def reduce(self, image, downscale=2, sigma=None, order=1,
                   mode='reflect', cval=0, multichannel=None):
        pass

class GaussianPyramid_CPU(GaussianPyramid):
    _strategy = ImplTypes.CPU_ONLY

    def __call__(self, data, level=None):
        pass

    def reduce(self):
        pass