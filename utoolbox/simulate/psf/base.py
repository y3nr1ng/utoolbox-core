from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np

Resolution = namedtuple('Resolution', ['dxy', 'dz'])

class PSF(object, metaclass=ABCMeta):
    def __init__(self, resolution=(1, 1)):
        if type(resolution) is not Resolution:
            try:
                resolution = Resolution._make(resolution)
            except TypeError:
                raise TypeError("invalid resolution input")
        self._resolution = resolution
        self._buffer = None

    @abstractmethod
    def __call__(self, wavelength):
        raise NotImplementedError

    @property
    def resolution(self):
        return self._resolution
