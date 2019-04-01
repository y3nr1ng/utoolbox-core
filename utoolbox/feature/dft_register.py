"""
Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

Port of Taylor Scott's code from skimage package:
https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/register_translation.py
"""
import logging
from math import ceil

import cupy as cp
import numpy as np

__all__ = [
    'DftRegister',
    'dft_register'
]

logger = logging.getLogger(__name__)

###
# region: Kernel definitions
###

ushort_to_float = cp.ElementwiseKernel(
    'uint16 src', 'float32 dst',
    'dst = (float)src',
    'ushort_to_float'
)

#cu_file = os.path.join(os.path.dirname(__file__), 'dft_register.cu')
#
#with open(cu_file, 'r') as fd:
#    source = fd.read()

###
# endregion
###

def _upsampled_dft(array, region_sz, upsample_factor=1, offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following operations are performed:
        - Embed the array to a larger one of size `upsample_factor` times larger in each dimension. 
        - ifftshift to bring the center of the image to (1, 1)
        - Take the FFT of the larger array.
        - Extract region of size [region_sz] from the result, starting with offsets.
    
    It achieves this result by computing the DFT in the output array without the need to zeropad. Much faster and memroy efficient than the zero-padded FFT approach if region_sz is much smaller than array.size * upsample_factor.

    :param cupy.ndarray array: DFT of the data to be upsampled
    :param int region_sz: size of the region to be sampled
    :param integer upsample_factor: the upsampling factor
    :param integers offsets: offsets to the sampling region
    :return: upsampled DFT of the specified region
    :rtype: cupy.ndarray
    """
    

def _compute_phase_diff(cross_corr_max):
    """
    Compute global phase difference betweeen the two images (should be zero if 
    images are non-negative).

    :param cupy.complex64 cross_corr_max: complex value of the cross correlation at its maximum point
    """
    return cp.arctan2(cross_corr_max.image, cross_corr_max.real)

@cp.fuse()
def _compute_error(cross_corr_max, template, target):
    """
    Compute RMS error metric between template, and target.

    :param cupy.complex64 cross_corr_max: complex value of the cross correlation at its maximum point
    :param np.float32 template: normalized average intensity of the template
    :param np.float32 target: normalized average intensity of the target
    """
    error = 1. - cross_corr_max * cross_corr_max.conj() / (template * target)
    return cp.sqrt(cp.abs(error))

class DftRegister(object):
    def __init__(self, template, upsample_factor=1):
        self._real_tpl, self._cplx_tpl = template, None
        self._upsample_factor = upsample_factor

    def __enter__(self):
        # upload to device
        _real_tpl = cp.empty(self.real_tpl.shape, dtype=cp.float32)
        ushort_to_float(cp.asarray(self.real_tpl), _real_tpl)
        self._real_tpl = _real_tpl
        
        # forward FT
        self._cplx_tpl = cp.fft.fft2(self.real_tpl)
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # restore and empty
        self._real_tpl = cp.asnumpy(self.real_tpl)
        self._cplx_tpl = None

    @property
    def cplx_tpl(self):
        return self._cplx_tpl
    
    @property
    def real_tpl(self):
        return self._real_tpl
    
    @property
    def upsample_factor(self):
        return self._upsample_factor

    def register(self, target, return_error=True):
        if target.shape != self.real_tpl.shape:
            raise ValueError("shape mismatch")

        ### DEVICE MEOMRY

        # upload target to device
        real_tar = cp.empty(target.shape, dtype=cp.float32)
        ushort_to_float(cp.asarray(target), real_tar)
        # forward FT
        cplx_tar = cp.fft.fft2(real_tar)

        # compute cross-correlation by IFT
        _product = self.cplx_tpl * cplx_tar.conj()
        cross_corr = cp.fft.ifft2(_product)

        # local maxima
        maxima = cp.unravel_index(
            cp.argmax(cp.abs(cross_corr)),
            cross_corr.shape
        )
        # wrap around
        shifts = np.array(maxima, dtype=np.float32)
        shape = np.array(target.shape)
        shifts[maxima > (shape//2)] -= shape

        upsample_factor = np.array(self.upsample_factor, dtype=np.float32)
        if upsample_factor == 1:
            if return_error:    
                raise NotImplementedError()
        else:
            # estimate shift by upsample factor
            shifts = np.round(shifts * upsample_factor) / upsample_factor
            region_sz = np.ceil(upsample_factor * 1.5, dtype=np.float32)

            # center the output array
            dft_shifts = np.floor(region_sz / 2., dtype=np.float32)
            logger.debug("dft_shifts={}".format(dft_shifts))

            region_offset = dft_shifts - shifts * upsample_factor
            logger.debug("region_offset={}".format(region_offset))
            
            # refine shift estimate by matrix multiply DFT
            cross_corr = _upsampled_dft(
                _product, 
                region_sz, 
                upsample_factor, 
                region_offset
            ).conj()

            #TODO normalize
            norm_factor = (self.cplx_tpl.size * upsample_factor ** 2)
            logger.debug("norm_factor.dtype={}".format(norm_factor.dtype))
            
            raise RuntimeError("DEBUG")

            cross_corr /= norm_factor

            #NOTE repeat!!!
            # local maxima
            maxima = cp.unravel_index(
                cp.argmax(cp.abs(cross_corr)),
                cross_corr.shape
            )
            # wrap around
            fine_shifts = np.array(maxima, dtype=np.float32)
            fine_shifts -= dft_shifts
            shifts += fine_shifts / upsample_factor

            logger.debug("shifts={}".format(shifts))

            raise RuntimeError("DEBUG")

            if return_error:
                raise NotImplementedError()

        tmp = cp.abs(cross_corr)
        from imageio import imwrite
        imwrite('debug.tif', cp.asnumpy(tmp))

        ### HOST MEMORY

def dft_register(image, template, upsample_factor=1, space='real', return_error=True):
    #TODO return_error?
    pass