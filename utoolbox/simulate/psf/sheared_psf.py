from math import radians, sin, cos, ceil

import numpy as np
import scipy.interpolate

from utoolbox.simulate.psf.base import PSF

#TODO inherit PSF, and override respective method while utilizing the `model`
class ShearedPSF(object):
    def __init__(self, model, angle, img_res, *args, **kwargs):
        """
        Parameters
        ----------
        angle : float
            Angle between sample stage and the detection objective in degrees.
        model : PSF
            The PSF model to use.
        img_res : tuple of float
            Resolution of the _acquired image_.

        Note
        ----
        PSF model resolution is automagically inferred.
        """
        if not issubclass(model, PSF):
            raise TypeError("invalid PSF model")

        # image coordinate, (z, y, x)
        angle = radians(angle)
        dxy, dz = img_res
        dk = dz * sin(angle)
        # psf coordinate, (k, j, i)
        psf_res = (dxy, dk)

        # initialize the PSF object
        kwargs['resolution'] = psf_res
        self._psf = model(*args, **kwargs)

        # sheared shifts per layer
        self._pixel_shift = dz * cos(angle)

    def __call__(self, shape, *args, **kwargs):
        # the completely sheared size
        nz, ny, nx = shape
        total_shift = ceil(self._pixel_shift * (nz-1))

        kwargs['mode'] = 'cylindrical'
        psf_zr, rv = self._psf((nz, ny, nx+total_shift), *args, **kwargs)

        # find origin
        x0, y0 = (nx-1)/2., (ny-1)/2.

        # generate Cartesian grid
        yxg = np.mgrid[0:ny, 0:nx]

        # radial interpolation
        psf_zyx = np.empty(shape, dtype=psf_zr.dtype)
        for iz in range(nz):
            psf_interp = scipy.interpolate.interp1d(rv, psf_zr[iz, :])
            # NOTE layered shift is reversed to simulate sample scan
            iz_shift = self._pixel_shift * (iz - (nz-1)/2.)
            rg = np.hypot(yxg[0]-y0, yxg[1]-x0+iz_shift) * self._psf.resolution.dxy
            psf_zyx[iz, :, :] = psf_interp(rg.ravel()).reshape(ny, nx)
        try:
            psf_zyx = np.squeeze(psf_zyx, axis=0)
        except ValueError:
            pass

        if self._psf.normalize == 'energy':
            psf_zyx /= np.sum(psf_zyx)

        return psf_zyx
