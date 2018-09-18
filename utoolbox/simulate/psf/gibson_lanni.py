"""
Generate a PSF using the Gibson and Lanni model.

References
----------
[1] Gibson, S. and Lanni, F. (1992). Experimental test of an analytical model of
    aberration in an oil-immersion objective lens used in three-dimensional
    light microscopy. Journal of the Optical Society of America A, 9(1), p.154.
[2] Fast and Accurate 3D PSF Computation for Fluorescence Microscopy. Available
    at: https://bit.ly/2PY7nHs
"""
from typing import NamedTuple
from math import hypot

import numpy as np
import scipy
import scipy.interpolate
import scipy.special

from .base import PSF

__all__ = [
    'FastGibsonLanni'
]

class FastGibsonLanni(PSF):
    """
    References
    ----------
    [1] Li, J., Xue, F. and Blu, T. (2017). Fast and accurate three-dimensional
        point spread function computation for fluorescence microscopy. Journal
        of the Optical Society of America A, 34(6), p.1029.
    [2] Douglass, K. (2018). Implementing a fast Gibson-Lanni PSF solver in
        Python. [online] Kyle M. Douglass. Available at: https://bit.ly/2CcfHAw
    """
    class Parameters(NamedTuple):
        # magnification
        M: int
        # numerical aperture
        NA: float
        # immersion medium refraction index, design value
        ni0: float
        # immersion medium refraction index, experimental value
        ni: float
        # specimen refractive index
        ns: float
        # working distance [um]
        ti0: float
        # particle distance from coverslip [um]
        zd0: float = 0.
        # coverslip refraction index, designed value
        ng0: float = 1.52
        # coverslip refraction index, experimental value
        ng: float = 1.52
        # coverslip thickness [um], design value
        tg0: float = 100.
        # coverslip thickness [um], experimental value
        tg: float = 100.


    # number of rescaled Bessels for phase function approximation
    n_basis = 200
    # number of pupil sample along the radial direction
    n_samples = 1000

    # minimum wavelength for series expansion [450, 700]
    min_wavelength = 0.350

    def __init__(self, parameters, has_coverslip=True, **kwargs):
        super(FastGibsonLanni, self).__init__(**kwargs)
        if type(parameters) is not FastGibsonLanni.Parameters:
            raise TypeError("invalid parameter type")
        self._parameters = parameters
        self._has_coverslip = has_coverslip

    @property
    def has_coverslip(self):
        return self._has_coverslip

    @property
    def parameters(self):
        return self._parameters

    def _generate_cartesian_profile(self, shape, wavelength, dtype, oversampling):
        psf_zr, rv = self._generate_cylindrical_profile(
            shape, wavelength, dtype, oversampling
        )

        nz, ny, nx = shape
        # find origin
        x0, y0 = (nx-1)/2., (ny-1)/2.

        # generate Cartesian grid
        yxg = np.mgrid[0:ny, 0:nx]
        rg = np.hypot(yxg[0]-y0, yxg[1]-x0) * self.resolution.dxy

        # radial interpolation
        psf_zyx = np.empty(shape, dtype=dtype)
        for iz in range(nz):
            psf_interp = scipy.interpolate.interp1d(rv, psf_zr[iz, :])
            psf_zyx[iz, :, :] = psf_interp(rg.ravel()).reshape(ny, nx)
        try:
            psf_zyx = np.squeeze(psf_zyx, axis=0)
        except ValueError:
            pass

        if self._normalize == 'energy':
            psf_zyx /= np.sum(psf_zyx)

        return psf_zyx

    def _generate_cylindrical_profile(self, shape, wavelength, dtype, oversampling):
        nz, ny, nx = shape
        # find origin
        x0, y0 = (nx-1)/2., (ny-1)/2.

        # find maximum radius, distance to the corner
        r_max = round(hypot(nx-x0, ny-y0)) + 1

        # polar coordinate, image space
        res = self.resolution
        rv = res.dxy * np.arange(0, oversampling*r_max, dtype=dtype) / oversampling

        # z steps
        zv = res.dz * (np.arange(-nz/2., nz/2., dtype=dtype) + .5)

        # wavefront aberrations
        W, rhov = self._opd(wavelength, zv, dtype)
        # chosen back focal plane aperture size
        a = rhov[-1]

        # sample the phase
        #   shape = (z steps, n_samples)
        phase = np.cos(W) + 1j*np.sin(W)

        # basis of Bessel functions
        na = self.parameters.NA
        scaling_factor = na * (3 * np.arange(1, FastGibsonLanni.n_basis+1, dtype=dtype) - 2) * FastGibsonLanni.min_wavelength / wavelength
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rhov)

        # approximation to the fourier space phase using LSE
        #   shape = (n_basis, z steps)
        C, residuals, _, _ = scipy.linalg.lstsq(J.T, phase.T)

        # convenient functions for J0 and J1
        J0 = lambda x: scipy.special.jv(0, x)
        J1 = lambda x: scipy.special.jv(1, x)

        k = 2*np.pi / wavelength
        beta = k * rv.reshape(-1, 1) * na
        denom = scaling_factor*scaling_factor - beta*beta
        R = (scaling_factor * J1(scaling_factor*a) * J0(beta*a) * a - beta * J0(scaling_factor*a) * J1(beta*a) * a)
        R /= denom

        psf_rz = (np.abs(R.dot(C))**2).T

        if self._normalize == 'peak':
            psf_rz /= np.max(psf_rz)

        return psf_rz, rv

    def _opd(self, wavelength, zv, dtype):
        # polar coorindate, fourier space
        p = self.parameters

        _a = [p.NA, p.ni0, p.ni, p.ns]
        if self.has_coverslip:
            _a += [p.ng0, p.ng]
        a = min(*_a) / p.NA
        rhov = np.linspace(0, a, FastGibsonLanni.n_samples, dtype=dtype)

        # immersion medium
        opd_i = (zv.reshape(-1, 1) + p.ti0) * np.sqrt(p.ni*p.ni - p.NA*p.NA * rhov*rhov) - p.ti0 * np.sqrt(p.ni0*p.ni0 - p.NA*p.NA * rhov*rhov)
        if self.has_coverslip:
            # coverslip
            opd_g = p.tg * np.sqrt(p.ng*p.ng - p.NA*p.NA * rhov*rhov) - p.tg0 * np.sqrt(p.ng0*p.ng0 - p.NA*p.NA * rhov*rhov)
            # sample
            opd_s = p.zd0 * np.sqrt(p.ns*p.ns - p.NA*p.NA * rhov*rhov)

            opd = opd_i + opd_g + opd_s
        else:
            opd = opd_i

        # define wavefront aberrations
        k = 2*np.pi / wavelength
        W = k * opd

        return W, rhov
