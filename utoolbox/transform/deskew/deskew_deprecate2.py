import logging
from math import hypot, sin, cos, radians, ceil
import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

logger = logging.getLogger(__name__)

class Deskew(metaclass=AbstractAlgorithm):
    """
    Specialized geometry transformation for LLSM datasets using sample scan.
    """
    def __call__(self, I, deg, res=(1., 1.), rotate=True, resample=True, 
    shape=None):
        """
        Parameters
        ----------
        I : np.ndarray
            Input image.
        deg : float
            Rotation angle in degrees.
        res : tuple of float
            Resolution in the form of (lateral, axial) axis, default is (1., 1.)
        rotate : boolean
            Rotate the output volume after shearing, default is True.
        resample : boolean
            Resample the volume to isotropic voxel size.
        shape : tuple of int, optional
            Output shape, default to maximize entire shape.

        Note
        ----
        Volume is sampled to isotropic scale.
        """
        if I.ndim != 3:
            raise NotImplementedError("only 3D data is allowed")
        nz, ny, nx = I.shape
        logger.debug("[input] x={}, y={}, z={}".format(nx, ny, nz))

        If = np.swapaxes(I, 0, 1)
        np.ascontiguousarray(If, dtype=np.float32) #TODO slow
        
        #If = I.astype(np.float32) #NOTE merged in np.ascontiguousarray
        rad = radians(deg)
        res, shift, rot, ratio = self._estimate_parameters(
            I, rad, res, rotate, resample
        )
        logger.debug("shift={:.4f}, rot={}, ratio={:.4f}".format(shift, rot, ratio))
        if shape is None:
            shape = self._estimate_out_shape(If, res, shift, rot, ratio)
            logger.info("estimated output shape {}".format(shape))
        
        out_buf = np.empty(shape, dtype=np.float32)

        self._run(If, shift, rot, rotate, out_buf)

        out_buf = np.swapaxes(out_buf, 0, 1)
        np.ascontiguousarray(out_buf, dtype=I.dtype)
        return out_buf
    
    def _estimate_parameters(self, I, rad, res, rotate, resample):
        """
        Non-rigid parameters, including shearing and rescaling.

        Parameters
        ----------
        rad : float 
            Rotation angle in radians.
        res : tuple of float
            Resolution in the form of (lateral, axial) axis.
        resample : boolean 
            Resample the volume to isotropic voxel size.

        Return
        ------
        res : tuple of float
            Updated resolution in the form of (lateral, axial) axis.
        shift : float
            Plane shift in unit pixels.
        rot : tuple of float
            Rotation matrix, (sin(a), cos(a)).
        ratio : float
            Ratio of axial resolution over lateral resolution.
        """
        lat, ax = res
        shift = ax*cos(rad) / lat

        ax = ax*sin(rad)

        if resample:
            ratio = ax/lat
            shift /= ratio # shrink the step size
        else:
            ratio = 1.

        if rotate:
            # estimate discrete sin/cos
            _, ny, _ = I.shape
            dx, dy = ceil(shift * (ny-1)), ceil(ratio * ny)
            h = hypot(dx, dy)
            vsin, vcos = dy/h, dx/h
        else:
            vsin, vcos = (0., 1.)

        return (lat, ax), shift, (vsin, vcos), ratio

    def _estimate_out_shape(self, I, res, shift, rot, ratio):
        nz, ny, nx = I.shape
  
        #
        # shear: xyz -> uvw
        #
        # update shape based on scaling ratio
        nv = ceil(ratio * ny)
        nu = nx + ceil(shift * (nv-1))
        logger.debug("[sheared] x={}, y={}, z={}".format(nu, nz, nv))
        
        #
        # rotate: uvw -> pqr
        #
        corners = [(1., 1.), (-1., 1.), (-1., -1.), (1., -1.)]
        vsin, vcos = rot
        fx = lambda x, y: x*nu/2. * vcos - y*nv/2. * vsin
        fy = lambda x, y: x*nu/2. * vsin + y*nv/2. * vcos
        np = [fx(*p) for p in corners]
        nq = [fy(*p) for p in corners]

        # TODO crop at the edges, not the full size
        nq = int(ceil(max(nq)-min(nq)))
        np = int(ceil(max(np)-min(np)))
        logger.debug("[rotated] x={}, y={}, z={}".format(np, nz, nq))

        return (nz, nq, np)

    @interface
    def _run(self, I, shift, rot, ratio, out):
        """
        Parameters
        ----------
        I : np.ndarray, dtype=np.float32
            Input image.
        shift : float
            Plane shift in unit pixels.
        rot : tuple of float
            Rotation matrix, (sin(a), cos(a)).
        ratio : float
            Ratio of axial resolution over lateral resolution.
        out : np.ndarray, dtype=np.float32
            Output array. Sampling grid is determined by size of this buffer.
        """
        pass

class Deskew_GPU(Deskew):
    _strategy = ImplTypes.GPU

    def __init__(self):
        # load kernel from file
        path = os.path.join(os.path.dirname(__file__), 'deskew.cu')
        with open(path, 'r') as fd:
            try:
                module = SourceModule(fd.read())
            except cuda.CompileError as err:
                logger.error("compile error: " + str(err))
                raise
        self._shear_kernel = module.get_function('shear_kernel')
        self._rotate_kernel = module.get_function('rotate_kernel')
        
        # preset texture
        shear_texture = module.get_texref('shear_tex')
        shear_texture.set_address_mode(0, cuda.address_mode.BORDER)
        shear_texture.set_address_mode(1, cuda.address_mode.BORDER)
        shear_texture.set_address_mode(2, cuda.address_mode.BORDER)
        shear_texture.set_filter_mode(cuda.filter_mode.LINEAR)
        self._shear_texture = shear_texture
        rotate_texture = module.get_texref('rotate_tex')
        rotate_texture.set_address_mode(0, cuda.address_mode.BORDER)
        rotate_texture.set_address_mode(1, cuda.address_mode.BORDER)
        rotate_texture.set_filter_mode(cuda.filter_mode.LINEAR)
        self._rotate_texture = rotate_texture

        # preset kernel launch parameters
        self._shear_kernel.prepare('PffIIffIII', texrefs=[shear_texture])
        self._rotate_kernel.prepare('PfIIffII', texrefs=[rotate_texture])

        # output staging buffer
        self._out_buf = None

    def _run(self, I, shift, rot, ratio, out):
        logger.debug("I.shape={}".format(I.shape))
        # bind input image to texture
        _in_buf = cuda.np_to_array(I, 'C')
        self._shear_texture.set_array(_in_buf)

        # determine grid and block size
        _, nv, nu = out.shape
        block_sz = (32, 32, 1)
        grid_sz = (ceil(float(nu)/block_sz[0]), ceil(float(nv)/block_sz[1]))

        if (self._out_buf is None) or (self._out_buf.shape != out.shape):
            logger.debug("resize buffer to {}".format(out.shape))
            self._out_buf = gpuarray.empty(out.shape, dtype=np.float32, order='C')

        # TODO create rotate kernel buffer area
            
        # execute
        nz, ny, nx = I.shape
        self._shear_kernel.prepared_call(
            grid_sz, block_sz,
            self._out_buf.gpudata,
            np.float32(shift),
            np.uint32(nu), np.uint32(nv),
            np.float32(ratio),
            np.uint32(nx), np.uint32(ny),
            np.float32(nz)
        )
        # TODO add rotate kernel call
        self._out_buf.get(out)

        # unbind texture
        _in_buf.free()
        