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
from collections import namedtuple
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
    Parameters = namedtuple('Parameters', [
        'M',    # magnification
        'NA',   # numerical aperture
        'ng0',  # coverslip refraction index, designed value
        'ng',   # coverslip refraction index, experimental value
        'ni0',  # immersion medium refraction index, design value
        'ni',   # immersion medium refraction index, experimental value
        'ns',   # specimen refractive index
        'ti0',  # working distance [um]
        'tg0',  # coverslip thickness [um], design value
        'tg',   # coverslip thickness [um], experimental value
        'zd0'   # tube length [um]
    ])

    # number of rescaled Bessels for phase function approximation
    n_basis = 200
    # number of pupil sample along the radial direction
    n_samples = 1000

    # minimum wavelength for series expansion
    min_wavelength = 0.350

    def __init__(self, parameters, **kwargs):
        super(FastGibsonLanni, self).__init__(**kwargs)
        if type(parameters) is not FastGibsonLanni.Parameters:
            raise TypeError("invalid parameter type")
        self.parameters = parameters

    def __call__(self, shape, wavelength, dtype=np.float32, oversampling=2):
        if len(shape) == 2:
            nz, (ny, nx) = 1, shape
        else:
            nz, ny, nx = shape

        if (self._buffer is None) or (self._buffer.shape != shape):
            self._buffer = np.empty(shape, dtype=dtype)

        # find origin
        x0, y0 = (nx-1)/2., (ny-1)/2.

        # find maximum radius, distance to the corner
        r_max = round(hypot(nx-x0, ny-y0)) + 1

        # polar coordinate, image space
        rv = self.resolution.dxy * np.arange(0, oversampling*r_max, dtype=dtype) / oversampling
        # polar coorindate, fourier space
        p = self.parameters
        a = min([x for x in [p.NA, p.ng0, p.ng, p.ni0, p.ni, p.ns] if x > 0.]) / p.NA
        rhov = np.linspace(0, a, FastGibsonLanni.n_samples, dtype=dtype)

        # z steps
        zv = self.resolution.dz * (np.arange(-nz/2., nz/2., dtype=dtype) + .5)

        # define wavefront aberrations
        # sample
        opd_s = p.zd0 * np.sqrt(p.ns*p.ns - p.NA*p.NA * rhov*rhov)
        # immesion medium
        opd_i = (zv.reshape(-1, 1) + p.ti0) * np.sqrt(p.ni*p.ni - p.NA*p.NA * rhov*rhov) - p.ti0 * np.sqrt(p.ni0*p.ni0 - p.NA*p.NA * rhov*rhov)
        # coverslip
        opd_g = p.tg * np.sqrt(p.ng*p.ng - p.NA*p.NA * rhov*rhov) - p.tg0 * np.sqrt(p.ng0*p.ng0 - p.NA*p.NA * rhov*rhov)
        # total
        k = 2*np.pi / wavelength
        #W = k * (opd_s + opd_i + opd_g)
        W = k * (opd_s + opd_i)

        # sample the phase
        #   shape = (z steps, n_samples)
        phase = np.cos(W) + 1j*np.sin(W)

        # basis of Bessel functions
        scaling_factor = p.NA * (3 * np.arange(1, FastGibsonLanni.n_basis+1, dtype=dtype) - 2) * FastGibsonLanni.min_wavelength / wavelength
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rhov)

        # approximation to the fourier space phase using LSE
        #   shape = (n_basis, z steps)
        C, residuals, _, _ = scipy.linalg.lstsq(J.T, phase.T)

        #return rhov, phase, J, C

        # convenient functions for J0 and J1
        J0 = lambda x: scipy.special.jv(0, x)
        J1 = lambda x: scipy.special.jv(1, x)

        beta = k * rv.reshape(-1, 1) * p.NA
        denom = scaling_factor*scaling_factor - beta*beta
        R = (scaling_factor * J1(scaling_factor*a) * J0(beta*a) * a - beta * J0(scaling_factor*a) * J1(beta*a) * a)
        R /= denom

        psf_rz = (np.abs(R.dot(C))**2).T

        # normalize
        psf_rz /= np.max(psf_rz)

        # generate Cartesian grid
        xyg = np.mgrid[0:ny, 0:nx]
        rg = np.sqrt((xyg[1]-x0)*(xyg[1]-x0) + (xyg[0]-y0)*(xyg[0]-y0)) * self.resolution.dxy

        psf_xyz = np.empty((nz, ny, nx), dtype=dtype)
        for iz in range(nz):
            psf_interp = scipy.interpolate.interp1d(rv, psf_rz[iz, :])
            psf_xyz[iz, :, :] = psf_interp(rg.ravel()).reshape(ny, nx)
        print("{}.dtype={}".format("psf_xyz", psf_xyz.dtype))

        return psf_xyz

    def _generate_grid(self, shape):
        pass

    def _generate_zyx_profile(self):
        pass

    def _generate_zr_profile(self, zv, wavelength):
        pass
