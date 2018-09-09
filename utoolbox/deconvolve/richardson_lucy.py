"""
The Richardson–Lucy algorithm is an iterative procedure for recovering an
underlying image that has been blurred by a known point spread function.
"""
import logging
from warnings import warn

import numpy as np
import pyopencl as cl

logger = logging.getLogger(__name__)

def is_optimal_size(n, factors=(2, 3, 5, 7, 11, 13)):
    n = int(n)
    assert n > 0, "size must be a positive integer"
    for factor in factors:
        while n % factor == 0:
            n /= factor
    return n == 1

def find_optimal_size(target, prefer_add=True):
    if is_optimal_size(target):
        return target
    else:
        for abs_delta in range(1, target):
            sign = 1 if prefer_add else -1
            for delta in (sign*abs_delta, -sign*abs_delta):
                candidate = target + delta
                if is_optimal_size(candidate):
                    return candidate

class RichardsonLucy(object):
    def __init__(self, context, shape, n_iter=10, prefer_add=False):
        self.n_iter = n_iter
        self._in_shape = tuple(shape)
        self._out_shape = tuple(
            [find_optimal_size(n, prefer_add=prefer_add) for n in shape]
        )

        # determine roi
        in_roi, out_roi = [], []
        for n_in, n_out in zip(self._in_shape, self._out_shape):
            dn = n_out - n_in
            if dn < 0:
                # output smaller then input
                in_roi.append(slice((-dn)//2, (-dn)//2 + n_out))
                out_roi.append(slice(0, n_out))
            elif dn > 0:
                # input smaller then output
                in_roi.append(slice(0, n_in))
                out_roi.append(slice(d//2, d//2 + n_in))
            else:
                in_roi.append(slice(0, n_in))
                out_roi.append(slice(0, n_out))
        in_roi, out_roi = tuple(in_roi), tuple(out_roi)
        self._crop_func = lambda ref, out: out[out_roi] = ref[in_roi]

        self.context = context

    def __enter__(self):
        self._allocate_workspace()

    def __exit__(self):
        self._free_workspace()

    def __call__(self, data):
        if data.shape != self._in_shape:
            warn("input size does not match the design specification")

        # copy to staging buffer

        # transfer


    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, new_n_iter):
        if new_n_iter < 1:
            raise ValueError("at least 1 iteration is required")
        self._n_iter = new_n_iter

    @property
    def shape(self):
        return self._shape

    def run(self, data):
        for i_iter in range(self.n_iter):
            logger.verbose("iter {}".format(i_iter+1))
            if i_iter > 2:
                # Andrew-Biggs, 2nd order prediction, clip [0, 1]
                factor = self._calculate_acceleration_factor()
            else:
                Y_k = X_k #TODO

            # move X_k to X_kminus1

            if i_iter > 1:
                # move G_kminus1 to G_kminus2

            # raw / blurred
            CC = Y_k #TODO
            self._filter(CC, no_conj)
            self.calc_lr_core(CC)

            # determine next iteration
            self._filter(CC, conj)
            self.update_curr_estimate()

            self.calc_curr_pref_diff()

    def _allocate_workspace(self):
        ### REF staging buffers between host and device ###
        self.h_buf = np.zeros(shape=(nv, nu), dtype=dtype)
        self.d_buf =

        in_nbytes = self._in_shape
        self.d_in = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE,
            size=self.h_in.nbytes
        )

        self.h_out = None

    def _free_workspace(self):
        pass
