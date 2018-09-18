from abc import ABCMeta, abstractmethod

import numpy as np

from utoolbox.container import Resolution

class PSF(object, metaclass=ABCMeta):
    def __init__(self, normalize='peak', resolution=(1, 1)):
        if type(resolution) is not Resolution:
            try:
                resolution = Resolution._make(resolution)
            except TypeError:
                raise TypeError("invalid resolution input")
        self._resolution = resolution
        self.normalize = normalize

    def __call__(self, shape, wavelength, dtype=np.float32, mode='cartesian',
                 oversampling=2):
        """
        Parameters
        ----------
        shape : tuple of int
            Size of the generated model, can be 2D or 3D.
        wavelength : float
            Target wavelength in microns.
        dtype : np.dtype
            Generated PSF array data type.
        mode : str
            Generated array, can be either 'cartesian' or 'cylindrical'.
        oversampling : float
            Oversampling ratio.
        """
        if len(shape) == 2:
            shape = (1, ) + shape

        modes = {
            'cartesian': self._generate_cartesian_profile,
            'cylindrical': self._generate_cylindrical_profile
        }
        return modes[mode](shape, wavelength, dtype, oversampling)

    @property
    def resolution(self):
        return self._resolution

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, strategy):
        if strategy not in ('none', 'energy', 'peak'):
            raise ValueError("invalid normalization strategy")
        self._normalize = strategy

    @abstractmethod
    def _generate_cartesian_profile(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _generate_cylindrical_profile(self, *args, **kwargs):
        raise NotImplementedError
