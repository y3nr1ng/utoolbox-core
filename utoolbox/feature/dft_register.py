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
    def __init__(self, template, upsample_factor=1):
        self._real_tpl, self._cplx_tpl = template, None

    def __enter__(self):
        self._cplx_tpl = cp.fft.fft2(self.real_tpl)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._cplx_tpl = None

    @property
    def cplx_tpl(self):
        return self._cplx_tpl
    
    @property
    def real_tpl(self):
        return self._real_tpl

    def register(self, real_img, return_error=True):
        if real_img.shape != self.real_tpl.shape:
            raise ValueError("shape mismatch")
        cplx_img = cp.fft.fft2(real_img)

        ### DEVICE MEOMRY

        # compute cross-correlation by IFT
        _product = self.cplx_tpl * cplx_img.conj()
        cross_corr = cp.fft.ifft2(_product)

        print(cross_corr)

        # find local minimum


        ### HOST MEMORY

def dft_register(image, template, upsample_factor=1, space='real', return_error=True):
    #TODO return_error?
    pass