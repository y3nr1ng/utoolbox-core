"""
Port of MPI-CBG PhaseCorrelation
"""

import cupy as cp
from cupyx.scipy.fftpack import get_fft_plan, fftn, ifftn
import numpy as np

from utoolbox.utils.decorator import timeit

from utoolbox.stitching.findpeak import FindPeak3D


class PhaseCorrelation(object):
    """
    TBA

    Args:
        image1 (array): reference image
        image2 (array): target image
        n_peaks (int): number of peaks to match 
        verify (bool): verify the result with cross correlation
    """

    def __init__(self, image1, image2, n_peaks=5, verify=True):
        self.image1 = image1
        self.image2 = image2
        self._n_peaks = n_peaks
        self._verify = verify

    ##

    @timeit
    def run(self):
        max_shape = self._find_max_shape()

        # compute FT, assuming they are the same size
        fft1 = cp.asarray(self.image1, dtype=cp.complex64)
        fft2 = cp.asarray(self.image2, dtype=cp.complex64)

        plan = get_fft_plan(fft1, value_type="C2C")
        fft1 = fftn(fft1, overwrite_x=True, plan=plan)
        fft2 = fftn(fft2, overwrite_x=True, plan=plan)

        print(f"shape: {fft1.shape}, dtype: {fft1.dtype}")

        @cp.fuse
        def normalize(fft_image):
            re, im = cp.real(fft_image), cp.imag(fft_image)
            length = cp.sqrt(re * re + im * im)
            return fft_image / length

        fft1 = normalize(fft1)
        fft2 = cp.conj(normalize(fft2))

        # phase correlation spectrum
        pcm = fft1 * fft2
        pcm = ifftn(pcm, overwrite_x=True, plan=plan)
        pcm = cp.real(pcm)

        from skimage.morphology import disk
        from skimage.filters import median
        pcm = cp.asnumpy(pcm)
        pcm = median(pcm, disk(3))
        pcm = cp.asarray(pcm)
        
        peak_list = self._extract_correlation_peaks(pcm)

    ##

    @property
    def n_peaks(self):
        return self._n_peaks

    @property
    def verify(self):
        return self._verify

    ##

    def _find_max_shape(self):
        return tuple(
            max(ax1, ax2) for ax1, ax2 in zip(self.image1.shape, self.image2.shape)
        )

    def _extract_correlation_peaks(self, pcm):
        finder = FindPeak3D()
        peak_list = finder(pcm)

        from pprint import pprint

        pprint(peak_list)
