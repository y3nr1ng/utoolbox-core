from math import radians, sin, cos, ceil

from utoolbox.simulate.psf.base import PSF

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
        nx += ceil(self._pixel_shift * (nz-1))

        kwargs['mode'] = 'cylindrical'
        psf_zr, rv = self._psf(shape, *args, **kwargs)

        # resample to sheared Cartesian coordinate system
