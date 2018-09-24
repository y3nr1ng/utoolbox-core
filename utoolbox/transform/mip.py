import logging
from math import ceil
import os

from jinja2 import Template
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.tools import dtype_to_ctype

logger = logging.getLogger(__name__)

class MIPTransform(object):
    def __init__(self, direction):
#        if direction not in ('x', 'y', 'z'):
        if direction not in ('z'):
            raise ValueError("invalid direction")
        self._direction = direction

        self._shape, self._dtype = None, None

    def __call__(self, data, copy=False):
        if (data.shape != self._shape) or (data.dtype != self._dtype):
            logger.info("resizing workspace")
            if self._shape is not None:
                self.destroy_workspace()
            self._shape, self._dtype = data.shape, data.dtype

            try:
                self._load_kernel()
            except cuda.CompileError:
                logger.error("compile error")
                raise

            self.create_workspace()

        nz, ny, nx = self._shape

        # determine grid & block size
        n_blocks = (32, 32, 1)
        n_grids = (ceil(float(nx)/n_blocks[0]), ceil(float(ny)/n_blocks[1]), 1)

        d_vol = gpuarray.to_gpu(data)
        self._kernel.prepared_call(
            n_grids, n_blocks,
            self._d_buf.gpudata,
            d_vol.gpudata,
            np.int32(nx), np.int32(ny), np.int32(nz)
        )

        return self._d_buf.get()

    @property
    def direction(self):
        return self._direction

    def create_workspace(self):
        _, ny, nx = self._shape
        self._d_buf = gpuarray.empty((ny, nx), dtype=self._dtype, order='C')
        logger.info("workspace allocated, {}, {}".format(self._d_buf.shape, self._dtype))

    def destroy_workspace(self):
        pass

    def _load_kernel(self):
        path = os.path.join(os.path.dirname(__file__), "mip.cu")
        with open(path, 'r') as fd:
            tpl = Template(fd.read())
            source = tpl.render(src_type=dtype_to_ctype(self._dtype))
            module = SourceModule(source)

        kernel_name = "{}_proj_kernel".format(self._direction)
        self._kernel = module.get_function(kernel_name)

        self._kernel.prepare('PPiii')
