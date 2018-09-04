from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np

Resolution = namedtuple('Resolution', ['dxy', 'dz'])

class PSF(object, metaclass=ABCMeta):
    def __init__(self, normalize='max', resolution=(1, 1)):
        if type(resolution) is not Resolution:
            try:
                resolution = Resolution._make(resolution)
            except TypeError:
                raise TypeError("invalid resolution input")
        self._resolution = resolution
        self.normalize = normalize

    @abstractmethod
    def __call__(self, shape, wavelength, dtype):
        raise NotImplementedError

    @property
    def resolution(self):
        return self._resolution

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, strategy):
        if strategy not in ('none', 'energy', 'max'):
            raise ValueError("invalid normalization strategy")
        self._normalize = strategy
