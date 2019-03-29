"""
Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

Port of Taylor Scott's code from skimage package:
https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/register_translation.py
"""
import logging

import cupy as cp
import numpy as np

__all__ = [
    'DftRegister',
    'dft_register'
]

logger = logging.getLogget(__name__)

###
# region: Kernel definitions
###

#cu_file = os.path.join(os.path.dirname(__file__), 'dft_register.cu')
#
#with open(cu_file, 'r') as fd:
#    source = fd.read()

###
# endregion
###

def _upsampled_dft():
    """
    Upsampled DFT by matrix multiplication.
    """
    pass

def _compute_phase_diff():
    pass

def _compute_error():
    pass

class DftRegister(object):
    def __init__(self):
        pass    

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

def dft_register(image, template, upsample_factor=1, space='real', return_error=True):
    #TODO return_error?
    pass