"""
The Richardson–Lucy algorithm is an iterative procedure for recovering an
underlying image that has been blurred by a known point spread function.
"""
import logging
from warnings import warn

from gpyfft import FFT
from gpyfft.gpyfftlib import CLFFT_SINGLE
from humanfriendly import format_size as format_byte_size
import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
import pyopencl.tools

from utoolbox.container import AttrDict

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
    def __init__(self, context, shape, prefer_add=False, n_iter=10):
        self.context = context

        self._in_shape = tuple(shape)
        self._out_shape = tuple(
            [find_optimal_size(n, prefer_add=prefer_add) for n in self._in_shape]
        )
        logger.info("shape: in={}, out={}".format(self._in_shape, self._out_shape))

        self.n_iter = n_iter

        # determine roi
        in_roi, out_roi = [], []
        for n_in, n_out in zip(self._in_shape, self._out_shape):
            dn = n_out - n_in
            if dn < 0:
                # smaller output
                in_roi.append(slice((-dn)//2, (-dn)//2 + n_out))
                out_roi.append(slice(0, n_out))
            elif dn > 0:
                # smaller input
                in_roi.append(slice(0, n_in))
                out_roi.append(slice(d//2, d//2 + n_in))
            else:
                in_roi.append(slice(0, n_in))
                out_roi.append(slice(0, n_out))
        in_roi, out_roi = tuple(in_roi), tuple(out_roi)
        def _crop_func(dst, src):
            dst[out_roi] = src[in_roi]
        self._crop_func = _crop_func

        self._estimate_func = ElementwiseKernel(
            self.context,
            "float *out, float *ref, float eps",
            "out[i] = ref[i] / ((out[i] > eps) ? out[i] : eps)",
            "estimate_func"
        )

    def __enter__(self):
        self.queue = cl.CommandQueue(self.context)
        self._allocate_workspace()
        return self

    def __exit__(self, *args):
        self._free_workspace()

    def __call__(self, data):
        if data.shape != self._in_shape:
            warn("input size does not match the design specification")

        # copy to staging buffer
        self.h_buf.fill(0.)
        self._crop_func(self.h_buf, data)
        self.d_ref.set(self.h_buf)
        # NOTE use implicit host-side staging buffer to separate the internal
        # np.float32 environment and unknown dtype inputs
        logger.debug("data uploaded")

        self.run(self.d_dec_bufs.tmp, self.d_ref)

        # copy from staging buffer
        self.d_dec_bufs.tmp.get(ary=self.h_buf)
        logger.debug("data downloaded")

        return self.h_buf.copy()

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, new_n_iter):
        if new_n_iter < 1:
            raise ValueError("at least 1 iteration is required")
        self._n_iter = new_n_iter

    @property
    def otf(self):
        try:
            return self.d_otf
        except AttributeError:
            raise RuntimeError("PSF is not specified")

    @property
    def psf(self):
        return self._psf

    @psf.setter
    def psf(self, data):
        if data.shape != self._in_shape:
            warn("input size does not match the design specification")
        self._psf = np.zeros(self._out_shape, dtype=np.float32)
        self._crop_func(self._psf, data)

        # generate OTF
        try:
            # borrow reference image buffer
            self.d_ref.set(self._psf)
            logger.debug("psf loaded")
        except AttributeError:
            raise RuntimeError("workspace not allocated")
        self.fft.enqueue_arrays(data=self.d_ref, result=self.d_otf)
        logger.debug("psf -> otf")

    @property
    def shape(self):
        return self._shape

    def run(self, dst_array, src_array):
        """
        init
        ... x_{k+1} = ref
        ... a_k = 0.

        iter n (y_k, x_{k+1}, x_k, g_{k+1}, g_k)
        ... y_k = (a_k + 1) * x_{k+1} - a_k * x_k
        ... t = f(x_{k+1})
        ... g_k = g_{k+1}
            g_{k+1} = t - y_k
        ... x_k = x_{k+1}
            x_{k+1} = y_k + g_{k+1}
        ... a_k = L(g_{k+1}, g_k)
        --> return x_{k+1}
        --> return t, bypass acceleration
        """
        eps = np.finfo(np.float32).eps

        # x_{k+1} = ref
        self.d_acc_bufs.x1[:] = src_array
        # a_k = 0
        a = 0.
        logger.debug("[acc] init x_{k+1}, a_k")

        # TODO use progress bar
        for i_iter in range(self.n_iter):
            logger.info("iter {}".format(i_iter+1))

            if i_iter > 1:
                # a_k = L(g_{k+1}, g_k)
                a_nom = cl.array.dot(self.d_acc_bufs.g1, self.d_acc_bufs.g0).get()
                a_den = cl.array.dot(self.d_acc_bufs.g0, self.d_acc_bufs.g0).get()
                logger.debug("... a_frac={:.5f}/{:.5f}".format(a_nom, a_den))
                a = np.float32(max(min(a_nom/(a_den + eps), 1.), 0.))
                logger.debug("[acc] a_k={:.5f}, updated".format(a))

            # x: iterated point
            # y: predicted point

            # y_k = (a_k + 1) * x_{k+1} - a_k * x_k
            self.d_acc_bufs.y = np.float32(a+1) * self.d_acc_bufs.x1 - a * self.d_acc_bufs.x0
            logger.debug("[acc] y_k updated")

            # x_k = x_{k+1}
            self.d_acc_bufs.x0, self.d_acc_bufs.x1 = self.d_acc_bufs.x1, self.d_acc_bufs.x0
            logger.debug("[acc] x_k updated")

            # x_{k+1} = f(x_k)
            self.run_once(self.d_acc_bufs.x1, self.d_acc_bufs.y)

            # g_k = g_{k+1}
            self.d_acc_bufs.g0, self.d_acc_bufs.g1 = self.d_acc_bufs.g1, self.d_acc_bufs.g0
            # g_{k+1} = x_{k+1} - y_k
            self.d_acc_bufs.g1 = self.d_acc_bufs.x1 - self.d_acc_bufs.y
            logger.debug("[acc] g_k, g_{k+1} updated")

        dst_array[:] = self.d_acc_bufs.x1

    def run_once(self, dst_array, src_array):
        nz, ny, nx = self._out_shape
        scale = np.float32(nx*ny*nz)
        eps = np.finfo(np.float32).eps

        # step 0
        #   blur by psf
        self.fft.enqueue_arrays(
            data=src_array, result=self.d_dec_bufs.fft,
            forward=True
        )
        self.d_dec_bufs.fft *= self.d_otf
        self.ifft.enqueue_arrays(
            data=self.d_dec_bufs.fft, result=self.d_dec_bufs.tmp,
            forward=False
        )
        self.d_dec_bufs.tmp /= scale
        logger.debug("[lr_core] step 0")

        # step 1
        #   estimate next iteration
        self._estimate_func(self.d_dec_bufs.tmp, self.d_ref, eps)
        logger.debug("[lr_core] step 1")

        # step 2
        #   re-blur
        self.fft.enqueue_arrays(
            data=self.d_dec_bufs.tmp, result=self.d_dec_bufs.fft,
            forward=True
        )
        self.d_dec_bufs.fft *= self.d_otf.conj()
        self.ifft.enqueue_arrays(
            data=self.d_dec_bufs.fft, result=self.d_dec_bufs.tmp,
            forward=False
        )
        self.d_dec_bufs.tmp /= scale
        logger.debug("[lr_core] step 2")

        # step 3
        #   multiply the factor
        dst_array[:] = src_array * self.d_dec_bufs.tmp
        logger.debug("[lr_core] step 3")

    def _allocate_workspace(self):
        """
        ... y_k = x_k + (a_k * h_k)
            h_k = x_k - x_{k-1}
        ... x_{k+1} = y_k + g_k
            g_k = f(y_k) - y_k
        ... a_k = \sum{g_{k-1} * g_{k-2}} / \sum{g_{k-2} * g_{k-2}}
            a_k \in (0, 1)

        init
        ... x_{k+1} = ref
        ... a_k = 0.

        iter 0
        ... h_k = None
            y_k = x_{k+1}
            y_k = (a_k + 1) * x_{k+1} - a_k * x_k
        ... t = f(x_k)
        ... g_k = t - y_k
            x_k = x_{k+1}
            x_{k+1} = y_k + g_k
        ... g_{k-2} = g_{k-1}
            g_{k-1} = g_k
        ... a_k = 0.

        iter 1
        ... h_k = x_{k+1} - x_k
            y_k = x_{k+1}
        ... t = f(x_{k+1})
        ... g_k = t - y_k
            x_k = x_{k+1}
            x_{k+1} = y_k + g_k
        ... g_{k-2} = g_{k-1}
            g_{k-1} = g_k
        ... a_k = L(g_{k-1}, g_{k-2})

        iter 2
        ... h_k = x_{k+1} - x_k
            y_k = x_{k+1} + a_k * h_k
        ... t = f(x_{k+1})
        ... g_k = t - y_k
            x_k = x_{k+1}
            x_{k+1} = y_k + g_k
        ... g_{k-2} = g_{k-1}
            g_{k-1} = g_k
        ... a_k = L(g_{k-1}, g_{k-2})

        iter n (y_k, x_{k+1}, x_k, g_{k+1}, g_k)
        ... y_k = (a_k + 1) * x_{k+1} - a_k * x_k
        ... t = f(x_{k+1})
        ... g_k = g_{k+1}
            g_{k+1} = t - y_k
        ... x_k = x_{k+1}
            x_{k+1} = y_k + g_{k+1}
        ... a_k = L(g_{k+1}, g_k)
        --> return x_{k+1}
        --> return t, bypass acceleration
        """
        # pre-calculate shapes
        nz, ny, nx = self._out_shape
        real_shape = (nz, ny, nx)
        complex_shape = (nz, ny, nx//2+1)

        # create memory pool
        allocator = cl.tools.ImmediateAllocator(
            self.queue, mem_flags=cl.mem_flags.READ_WRITE
        )
        self._mem_pool = cl.tools.MemoryPool(allocator)

        #TODO wrap this section in ExitStack, callback(_free_workspace)

        # reference image
        self.h_buf = np.empty(real_shape, dtype=np.float32)
        self.d_ref = cl.array.empty(
            self.queue, real_shape, np.float32, allocator=self._mem_pool
        )

        # otf
        self.d_otf = cl.array.empty(
            self.queue, complex_shape, np.complex64, allocator=self._mem_pool
        )

        # deconvolution io buffers
        self.d_dec_bufs = AttrDict()
        self.d_dec_bufs['tmp'] = cl.array.empty(
            self.queue, real_shape, np.float32, allocator=self._mem_pool
        )
        self.d_dec_bufs['fft'] = cl.array.empty(
            self.queue, complex_shape, np.complex64, allocator=self._mem_pool
        )

        # deconvolution fft/ifft plans
        self.fft = FFT(
            self.context, self.queue,
            self.d_dec_bufs.tmp,
            out_array=self.d_dec_bufs.fft
        )
        logger.debug(
            "fft buffer size: {}".format(
                format_byte_size(self.fft.plan.temp_array_size, binary=True)
            )
        )

        self.ifft = FFT(
            self.context, self.queue,
            self.d_dec_bufs.fft,
            out_array=self.d_dec_bufs.tmp,
            real=True
        )
        logger.debug(
            "ifft buffer size: {}".format(
                format_byte_size(self.ifft.plan.temp_array_size, binary=True)
            )
        )

        # accelerator buffers
        self.d_acc_bufs = AttrDict()
        for name in ('y', 'x1', 'x0', 'g1', 'g0'):
            self.d_acc_bufs[name] = cl.array.empty(
                self.queue, real_shape, np.float32, allocator=self._mem_pool
            )

        logger.debug(
            "held={}, active={}".format(
                self._mem_pool.held_blocks, self._mem_pool.active_blocks
            )
        )

    def _free_workspace(self):
        # free reference image
        self.d_ref.base_data.release()

        # free deconvolution buffers
        for buffer in self.d_dec_bufs.values():
            buffer.base_data.release()

        # free accelerator buffers
        for buffer in self.d_acc_bufs.values():
            buffer.base_data.release()

        # destroy memory pool
        self._mem_pool.stop_holding()
