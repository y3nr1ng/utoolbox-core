"""
The Richardson–Lucy algorithm is an iterative procedure for recovering an
underlying image that has been blurred by a known point spread function.
"""
import numpy as np
import pyopencl as cl


class RichardsonLucy(object):
    def __init__(self, n_iter=10, shape=None):
        self.n_iter = n_iter
        self._shape = shape

    def __call__(self, data):
        nz, ny, nx = data.shape
        pass

    @property
    def niter(self):
        return self._niter

    @niter.setter
    def n_iter(self, new_n_iter):
        if new_n_iter < 1:
            raise ValueError("at least 1 iteration is required")
        self._n_iter = new_n_iter

    @property
    def shape(self):
        return self._shape
