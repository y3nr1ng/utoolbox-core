import logging
from math import sin, cos, radians, ceil
import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

logger = logging.getLogger(__name__)

class Rotate2(metaclass=AbstractAlgorithm):
    def __call__(self, I, deg, scale=(1., 1.), out=None):
        """
        Parameters
        ----------
        I : np.ndarray
            Input image.
        deg : float
            Rotation angle in degrees.
        scale : tuple of float
            Scaling ratio of (X, Y) axis, default is (1., 1.)
        out : np.ndarray, optional
            Output buffer. If defined, output size is cropped to its size.
        """
        logger.debug("out.shape: {}".format(out.shape))
        if out is None:
            out_sz = I.shape
            # TODO recalculate output size
            #out_sz = (out_sz[0]*scale[1], out_sz[1]*scale[0])
            out_buf = np.empty(out_sz, dtype=I.dtype)
        else:
            out_buf = out

        rad = radians(deg)

        self._run(I, rad, scale, out_buf)

        return out_buf
    
    @interface
    def _run(self, I, rad, scale, out):
        """
        Parameters
        ----------
        I : np.ndarray
            Input image.
        rad : float
            Rotation angle in radians.
        scale : tuple of float
            Scaling ratio of (X, Y) axis, default is (1., 1.)
        out : np.ndarray
            Output buffer. Output sampling grid is determined by size of this 
            buffer.
        """
        pass

class Rotate2_GPU(Rotate2):
    _strategy = ImplTypes.GPU

    def __init__(self):
        # load kernel from file
        path = os.path.join(os.path.dirname(__file__), 'rotate.cu')
        with open(path, 'r') as fd:
            try:
                module = SourceModule(fd.read())
            except cuda.CompileError as err:
                logger.error("compile error: " + str(err))
                raise
        self._kernel = module.get_function('rot2_kernel')
        
        # preset texture
        texture = module.get_texref('rot2_tex')
        #texture.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        texture.set_address_mode(0, cuda.address_mode.BORDER)
        texture.set_address_mode(1, cuda.address_mode.BORDER)
        texture.set_filter_mode(cuda.filter_mode.LINEAR)
        logger.debug("texture flags: {}".format(texture.get_flags()))
        self._texture = texture

        # preset kernel launch parameters
        self._kernel.prepare('PfIIffII', texrefs=[texture])

        # output staging buffer
        self._out_buf = None

    def _run(self, I, rad, scale, out):
        # bind input image to texture
        _in_buf = cuda.np_to_array(I, 'C')
        self._texture.set_array(_in_buf)

        if (self._out_buf is None) or (self._out_buf.shape != out.shape):
            logger.debug("resize buffer to {}".format(out.shape))
            self._out_buf = gpuarray.empty(out.shape, dtype=np.float32, order='C')
            
        # determine grid and block size
        nv, nu = out.shape
        logger.debug("out.shape={}".format(out.shape))
        block_sz = (16, 16, 1)
        grid_sz = (ceil(float(nu)/block_sz[0]), ceil(float(nv)/block_sz[1]))
        logger.debug("grid_sz={}".format(grid_sz))

        # execute
        ny, nx = I.shape
        sx, sy = scale
        logger.debug(self._out_buf.dtype)
        self._kernel.prepared_call(
            grid_sz, block_sz,
            self._out_buf.gpudata,
            np.float32(rad),
            np.uint32(nu), np.uint32(nv),
            np.float32(sx), np.float32(sy),
            np.uint32(nx), np.uint32(ny)
        )
        self._out_buf.get(out)
        logger.debug("out: {}".format(out))

        # unbind texture
        _in_buf.free()
        