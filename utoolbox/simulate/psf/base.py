from abc import ABCMeta
from collections import namedtuple

import numpy as np

Resolution = namedtuple('Resolution', ['dz', 'dy', 'dx'])

class PSF(object, metaclass=ABCMeta):
    def __init__(self, shape, resolution=(1, 1, 1), dtype=np.float32):
        if type(resolution) is not Resolution:
            try:
                ndim = len(resolution)
                if ndmi == 2:
                    resolution = tuple([1] + list(resolution))
                resolution = Resolution._make(resolution)
            except TypeError:
                raise TypeError("invalid resolution")
        self._resolution = resolution
        
        self._buffer = np.empty(shape, dtype=dtype)

    @abstractmethod
    def __call__(self, wavelength):
        raise NotImplementedError

    @property
    def resolution(self):
        return self._resolution

    @property
    def shape(self):
        return self._buffer.shape
