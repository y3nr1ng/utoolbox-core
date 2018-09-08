"""
The Richardson–Lucy algorithm is an iterative procedure for recovering an
underlying image that has been blurred by a known point spread function.
"""
import numpy as np
import pyopencl as cl

def is_optimal_size(n, factors=(2, 3, 5, 7)):
    n = int(n)
    assert n > 0, "size must be a positive integer"
    for factor in factors:
        while n % factor == 0:
            n /= factor
    return n == 1

def find_optimal_size(target, prefer_pos=True):
    if is_optimal_size(target):
        return target
    else:
        for abs_delta in range(1, target):
            sign = 1 if prefer_pos else -1
            for delta in (sign*abs_delta, -sign*abs_delta):
                candidate = target + delta
                if is_optimal_size(candidate):
                    return candidate

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
