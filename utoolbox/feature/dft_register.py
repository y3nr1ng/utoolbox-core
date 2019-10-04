"""
Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

Port of Taylor Scott's code from skimage package:
https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/register_translation.py
"""
import logging
from math import ceil, floor

import cupy as cp
import numpy as np

__all__ = ["DftRegister", "dft_register"]

logger = logging.getLogger(__name__)

###
# region: Kernel definitions
###

ushort_to_float = cp.ElementwiseKernel(
    "uint16 src", "float32 dst", "dst = (float)src", "ushort_to_float"
)

###
# endregion
###


class DftRegister(object):
    def __init__(
        self, template, upsample_factor=1, return_error=True
    ):
        self._real_tpl, self._cplx_tpl = template, None
        self._upsample_factor = int(upsample_factor)
        self._return_error, self._int_tpl = return_error, None

    def __enter__(self):
        # upload to device
        _real_tpl = cp.empty(self.real_tpl.shape, dtype=cp.float32)
        ushort_to_float(cp.asarray(self.real_tpl), _real_tpl)
        self._real_tpl = _real_tpl

        # forward FT
        self._cplx_tpl = cp.fft.fft2(self.real_tpl)

        # normalization factor
        self._norm_factor = self.cplx_tpl.size * self.upsample_factor ** 2

        # intensity of the template
        if self._return_error:
            if self.upsample_factor > 1:
                # using upsampled intensity
                self._int_tpl = cp.asnumpy(
                    self._upsampled_dft(self.cplx_tpl * self.cplx_tpl.conj(), 1)
                )[0, 0]
                self._int_tpl /= np.float32(self.norm_factor)
            else:
                self._int_tpl = cp.asnumpy(cp.sum(cp.abs(self.cplx_tpl) ** 2))
                self._int_tpl /= self.cplx_tpl.size

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # restore and empty
        self._real_tpl = cp.asnumpy(self.real_tpl)
        self._cplx_tpl = None

    @property
    def cplx_tpl(self):
        return self._cplx_tpl

    @property
    def norm_factor(self):
        return self._norm_factor

    @property
    def real_tpl(self):
        return self._real_tpl

    @property
    def upsample_factor(self):
        return self._upsample_factor

    def register(self, target, return_error=True):
        if target.shape != self.real_tpl.shape:
            raise ValueError("shape mismatch")

        # upload target to device
        real_tar = cp.empty(target.shape, dtype=cp.float32)
        ushort_to_float(cp.asarray(target), real_tar)
        # forward FT
        cplx_tar = cp.fft.fft2(real_tar)

        # compute cross-correlation by IFT
        _product = self.cplx_tpl * cplx_tar.conj()
        cross_corr = cp.fft.ifft2(_product)

        # local maxima
        maxima = np.unravel_index(
            cp.asnumpy(cp.argmax(cp.abs(cross_corr))), target.shape
        )
        # coarse shifts, wrap around
        shifts = np.array(
            [
                (ax_sh - ax_sz) if ax_sh > ax_sz / 2.0 else ax_sh
                for ax_sh, ax_sz in zip(maxima, target.shape)
            ]
        )
        logger.debug("coarse_shifts={}".format(shifts))

        if self.upsample_factor == 1:
            if return_error:
                cross_corr_max = cp.asnumpy(cross_corr[maxima])
                int_tar = cp.asnumpy(cp.sum(cp.abs(target) ** 2))
                int_tar /= target.size
                error = self._compute_error(cross_corr_max, int_tar)
        else:
            region_sz = ceil(self.upsample_factor * 1.5)

            # center the output array at dft_shift+1
            dft_shift = region_sz // 2
            region_offset = tuple(
                dft_shift - shift * self.upsample_factor for shift in shifts
            )

            # refine shift estimate by matrix multiply DFT
            cross_corr = self._upsampled_dft(_product, region_sz, region_offset)
            # normalization
            cross_corr /= self.norm_factor

            # local maxima
            maxima = np.unravel_index(
                cp.asnumpy(cp.argmax(cp.abs(cross_corr))), cross_corr.shape
            )

            # wrap around
            shifts = tuple(
                shift + float(ax_max - dft_shift) / self.upsample_factor
                for shift, ax_max in zip(shifts, maxima)
            )
            logger.debug("fine_shifts={}".format(shifts))

            if return_error:
                cross_corr_max = cp.asnumpy(cross_corr[maxima])
                int_tar = cp.asnumpy(
                    self._upsampled_dft(cplx_tar * cplx_tar.conj(), 1)
                )[0, 0]
                int_tar /= self.norm_factor
                error = self._compute_error(cross_corr_max, int_tar)

        if return_error:
            return shifts, error
        else:
            return shifts

    def _compute_error(self, cross_corr_max, int_tar):
        """
        Compute RMS error metric between template, and target.

        Args:
            cross_corr_max (np.complex64): complex value of the cross correlation at its maximum point
            int_tar (np.complex64): normalized maximum intensity of the target array
        """
        error = 1.0 - cross_corr_max * cross_corr_max.conj() / (self._int_tpl * int_tar)
        return np.sqrt(np.abs(error))

    def _upsampled_dft(self, array, region_sz, offsets=None):
        """
        Upsampled DFT by matrix multiplication.

        This code is intended to provide the same result as if the following operations are performed:
            - Embed the array to a larger one of size `upsample_factor` times larger in each dimension. 
            - ifftshift to bring the center of the image to (1, 1)
            - Take the FFT of the larger array.
            - Extract region of size [region_sz] from the result, starting with offsets.
        
        It achieves this result by computing the DFT in the output array without the need to zeropad. Much faster and memroy efficient than the zero-padded FFT approach if region_sz is much smaller than array.size * upsample_factor.

        Args:
            array (cp.ndarray): DFT of the data to be upsampled
            region_sz (int or tuple of int): size of the region to be sampled
            offsets (int or tuple of int): offsets to the sampling region
        
        Returns:
            (cp.ndarray): upsampled DFT of the specified region
        """
        try:
            if len(region_sz) != array.ndim:
                raise ValueError("upsampled region size must match array dimension")
        except TypeError:
            # expand integer to list
            region_sz = (region_sz,) * array.ndim

        if offsets is None:
            offsets = (0,) * array.ndim
        else:
            if len(offsets) != array.ndim:
                raise ValueError("axis offsets must match array dimension")

        dim_props = zip(reversed(array.shape), reversed(region_sz), reversed(offsets))
        for ax_sz, up_ax_sz, ax_offset in dim_props:
            # float32 sample frequencies
            fftfreq = (
                cp.hstack(
                    (
                        cp.arange(0, (ax_sz - 1) // 2 + 1, dtype=cp.float32),
                        cp.arange(-(ax_sz // 2), 0, dtype=cp.float32),
                    )
                )
                / ax_sz
                / self.upsample_factor
            )
            # upsampling kernel
            kernel = cp.exp(
                (1j * 2 * np.pi)
                * (cp.arange(up_ax_sz, dtype=np.float32) - ax_offset)[:, None]
                * fftfreq
            )
            # convolve
            array = cp.tensordot(kernel, array, axes=(1, -1))
        return array


def dft_register(template, target, upsample_factor=1):
    with DftRegister(template, upsample_factor=upsample_factor) as dft_reg:
        return dft_reg.register(target, return_error=False)
