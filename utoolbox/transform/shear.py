import logging
from math import ceil
import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

logger = logging.getLogger(__name__)

class Shear2(metaclass=AbstractAlgorithm):
    def __call__(self, I, shift, scale=(1., 1.), shape=None):
        """
        Parameters
        ----------
        I : np.ndarray
            Input image.
        shift : tuple of float
            Shift factor in (X, Y) axis.
        scale : tuple of float
            Scaling ratio of (X, Y) axis, default is (1., 1.)
        shape : tuple of int, optional
            Output shape, default to maximize entire shape.
        """
        If = I.astype(np.float32)
        if shape is None:
            shape = self._estimate_out_shape(I, shift, scale)
            logger.debug("estimated output shape {}".format(shape))
        out_buf = np.empty(shape, dtype=np.float32)

        self._run(If, shift, scale, out_buf)

        return out_buf.astype(I.dtype)

    def _estimate_out_shape(self, I, shift, scale):
        #TODO
        ny, nx = I.shape
        
        # actual scale of the image
        sx, sy = scale
        nx *= sx
        ny *= sy
        
        # transform matrix
        dx, dy = shift
        fx = lambda x, y: x*nx/2. + dx * y*ny/2.
        fy = lambda x, y: y*ny/2. + dy * x*nx/2.

        # result corners
        corners = [(1., 1.), (-1., 1.), (-1., -1.), (1., -1.)]
        nu = [fx(*p) for p in corners]
        nv = [fy(*p) for p in corners]

        return (int(ceil(max(nv)-min(nv))), int(ceil(max(nu)-min(nu))))
    
    @interface
    def _run(self, I, shift, scale, out):
        """
        Parameters
        ----------
        I : np.ndarray, dtype=np.float32
            Input image.
        shift : tuple of float
            Shift factor in (X, Y) axis.
        scale : tuple of float
            Scaling ratio of (X, Y) axis, default is (1., 1.)
        out : np.ndarray, dtype=np.float32
            Output buffer. Output sampling grid is determined by size of this 
            buffer.
        """
        pass

class Shear2_GPU(Shear2):
    _strategy = ImplTypes.GPU

    def __init__(self):
        # load kernel from file
        path = os.path.join(os.path.dirname(__file__), 'shear.cu')
        with open(path, 'r') as fd:
            try:
                module = SourceModule(fd.read())
            except cuda.CompileError as err:
                logger.error("compile error: " + str(err))
                raise
        self._kernel = module.get_function('shear2_kernel')
        
        # preset texture
        texture = module.get_texref('shear2_tex')
        texture.set_address_mode(0, cuda.address_mode.BORDER)
        texture.set_address_mode(1, cuda.address_mode.BORDER)
        texture.set_filter_mode(cuda.filter_mode.LINEAR)
        self._texture = texture

        # preset kernel launch parameters
        self._kernel.prepare('PffIIffII', texrefs=[texture])

        # output staging buffer
        self._out_buf = None

    def _run(self, I, shift, scale, out):
        # bind input image to texture
        _in_buf = cuda.np_to_array(I, 'C')
        self._texture.set_array(_in_buf)

        if (self._out_buf is None) or (self._out_buf.shape != out.shape):
            logger.debug("resize buffer to {}".format(out.shape))
            self._out_buf = gpuarray.empty(out.shape, dtype=np.float32, order='C')
            
        # determine grid and block size
        nv, nu = out.shape
        block_sz = (32, 32, 1)
        grid_sz = (ceil(float(nu)/block_sz[0]), ceil(float(nv)/block_sz[1]))

        # execute
        ny, nx = I.shape
        dx, dy = shift
        sx, sy = scale
        self._kernel.prepared_call(
            grid_sz, block_sz,
            self._out_buf.gpudata,
            np.float32(dx), np.float32(dy),
            np.uint32(nu), np.uint32(nv),
            np.float32(sx), np.float32(sy),
            np.uint32(nx), np.uint32(ny)
        )
        self._out_buf.get(out)

        # unbind texture
        _in_buf.free()
        