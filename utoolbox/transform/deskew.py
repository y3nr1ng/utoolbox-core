import logging
<<<<<<< Updated upstream
import math
=======
<<<<<<< Updated upstream
from math import radians, sin, cos, ceil, hypot
=======
import math
>>>>>>> Stashed changes
>>>>>>> Stashed changes
import os

import numpy as np
from pycuda.compiler import SourceModule
<<<<<<< Updated upstream

=======
<<<<<<< Updated upstream
>>>>>>> Stashed changes
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

    def _load_kernel(self):
        path = os.path.join(os.path.dirname(__file__), "deskew.cu")
        with open(path, 'r') as fd:
            tpl = Template(fd.read())
            source = tpl.render(dst_type=dtype_to_ctype(self._dtype))
            module = SourceModule(source)

        self._kernel = module.get_function("deskew_kernel")

        self._d_px_shift, _ = module.get_global('px_shift')
        self._d_vsin, _ = module.get_global('vsin')
        self._d_vcos, _ = module.get_global('vcos')

        self._texture = module.get_texref("ref_vol")
        self._texture.set_address_mode(0, cuda.address_mode.BORDER)
        self._texture.set_address_mode(1, cuda.address_mode.BORDER)
        self._texture.set_address_mode(2, cuda.address_mode.BORDER)
        self._texture.set_filter_mode(cuda.filter_mode.LINEAR)

        self._kernel.prepare('Piiiii',texrefs=[self._texture])

    def _upload_ref_vol(self, data):
        """Upload the reference volume into texture memory."""
        assert data.dtype == np.float32, "np.float32 is required"
        ref_vol = cuda.np_to_array(data, 'C')
        self._texture.set_array(ref_vol)
        return ref_vol

import pycuda.driver as driver

logger = logging.getLogger(__name__)

class Deskew(object):
    def __init__(self, angle=32.8, direction='forward', resolution=(.102, .3), rotate=True):
        self._angle = angle
        self._direction = direction
        self._resolution = resolution
        self._rotate = rotate

        self._out_buffer = None

        self._copy = self._load_copy_kernel()
        self._transpose = self._load_transpose_kernel()

    @property
    def angle(self):
        return math.degrees(self._angle)
    
    @angle.setter
    def angle(self, degree):
        self._angle = math.radians(degree)

    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, direction):
        if direction in ('forward', 'reversed'):
            self._direction = direction
        else:
            raise ValueError("invalid shear direction")

    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution

    @property
    def rotate(self):
        return self._rotate
    
    @rotate.setter
    def rotate(self, rotate):
        self._rotate = rotate

    def run(self, array):
        shape = array.shape
        if len(shape) != 3:
            raise ValueError("only 3D array is allowed")
        
        nz, ny, nx = shape
        logger.debug("shape, nx={}, ny={}, nz={}".format(nx, ny, nz))

        self._prepare_workspace(array.dtype)

        locked_in_array = driver.pagelocked_empty_like(array)
        locked_in_array[:] = array

        locked_out_array = self._copy(locked_in_array)
        
        out_array = np.empty_like(locked_out_array)
        out_array[:] = locked_out_array
        
        return out_array

    def _load_copy_kernel(self):
        path = os.path.join(os.path.dirname(__file__), 'deskew.cu')
        with open(path, 'r') as fd:
            module = SourceModule(fd.read())
        
        kernel = module.get_function('copy_kernel')

        def function(in_array):
            nz, ny, nx = in_array.shape
            out_array = driver.pagelocked_empty_like(
                in_array,
                mem_flags=driver.host_alloc_flags.DEVICEMAP
            )

            block_size = (32, 32, 1)
            grid_size = (-(-nx//32), -(-ny//32), 1)

            kernel(
                driver.Out(out_array),
                driver.In(in_array),
                np.int32(nx), np.int32(ny), np.int32(nz),
                grid=grid_size, block=block_size
            )

            return out_array
        
        return function

    def _load_transpose_kernel(self, tile_size=16, elements_per_thread=4):
        path = os.path.join(os.path.dirname(__file__), 'deskew.cu')
        with open(path, 'r') as fd:
            module = SourceModule(fd.read())
        
        kernel = module.get_function('transpose_xzy_outofplace')

        def function(in_array):
            nz, ny, nx = in_array.shape
            out_array = driver.pagelocked_empty(
                (ny, nz, nx), 
                in_array.dtype, 
                mem_flags=driver.host_alloc_flags.DEVICEMAP
            )

            block_size = (
                tile_size,
                tile_size//elements_per_thread,
                1
            )
            grid_size = (
                -(-nx//tile_size),
                -(-ny//tile_size),
                1
            )
            logger.debug("grid={}, block={}".format(grid_size, block_size))

            kernel(
                driver.Out(out_array), 
                driver.In(in_array),
                np.int32(nx), np.int32(ny), np.int32(nz),
                grid=grid_size, block=block_size
            )

            return out_array
        
        return function

    def _prepare_workspace(self, dtype):
        out_shape = self._estimate_output_shape()


    def _upload_ref_vol(self, data):
        """Upload the reference volume into texture memory."""
        assert data.dtype == np.float32, "np.float32 is required"
        ref_vol = cuda.np_to_array(data, 'C')
        self._texture.set_array(ref_vol)
        return ref_vol
<<<<<<< Updated upstream

=======
=======
>>>>>>> Stashed changes
import pycuda.driver as driver

logger = logging.getLogger(__name__)

class Deskew(object):
    def __init__(self, angle=32.8, direction='forward', resolution=(.102, .3), rotate=True):
        self._angle = angle
        self._direction = direction
        self._resolution = resolution
        self._rotate = rotate

        self._out_buffer = None

        self._copy = self._load_copy_kernel()
        self._transpose = self._load_transpose_kernel()

    @property
    def angle(self):
        return math.degrees(self._angle)
    
    @angle.setter
    def angle(self, degree):
        self._angle = math.radians(degree)

    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, direction):
        if direction in ('forward', 'reversed'):
            self._direction = direction
        else:
            raise ValueError("invalid shear direction")

    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution

    @property
    def rotate(self):
        return self._rotate
    
    @rotate.setter
    def rotate(self, rotate):
        self._rotate = rotate

    def run(self, array):
        shape = array.shape
        if len(shape) != 3:
            raise ValueError("only 3D array is allowed")
        
        nz, ny, nx = shape
        logger.debug("shape, nx={}, ny={}, nz={}".format(nx, ny, nz))

        self._prepare_workspace(array.dtype)

        locked_in_array = driver.pagelocked_empty_like(array)
        locked_in_array[:] = array

        locked_out_array = self._copy(locked_in_array)
        
        out_array = np.empty_like(locked_out_array)
        out_array[:] = locked_out_array
        
        return out_array

    def _load_copy_kernel(self):
        path = os.path.join(os.path.dirname(__file__), 'deskew.cu')
        with open(path, 'r') as fd:
            module = SourceModule(fd.read())
        
        kernel = module.get_function('copy_kernel')

        def function(in_array):
            nz, ny, nx = in_array.shape
            out_array = driver.pagelocked_empty_like(
                in_array,
                mem_flags=driver.host_alloc_flags.DEVICEMAP
            )

            block_size = (32, 32, 1)
            grid_size = (-(-nx//32), -(-ny//32), 1)

            kernel(
                driver.Out(out_array),
                driver.In(in_array),
                np.int32(nx), np.int32(ny), np.int32(nz),
                grid=grid_size, block=block_size
            )

            return out_array
        
        return function

    def _load_transpose_kernel(self, tile_size=16, elements_per_thread=4):
        path = os.path.join(os.path.dirname(__file__), 'deskew.cu')
        with open(path, 'r') as fd:
            module = SourceModule(fd.read())
        
        kernel = module.get_function('transpose_xzy_outofplace')

        def function(in_array):
            nz, ny, nx = in_array.shape
            out_array = driver.pagelocked_empty(
                (ny, nz, nx), 
                in_array.dtype, 
                mem_flags=driver.host_alloc_flags.DEVICEMAP
            )

            block_size = (
                tile_size,
                tile_size//elements_per_thread,
                1
            )
            grid_size = (
                -(-nx//tile_size),
                -(-ny//tile_size),
                1
            )
            logger.debug("grid={}, block={}".format(grid_size, block_size))

            kernel(
                driver.Out(out_array), 
                driver.In(in_array),
                np.int32(nx), np.int32(ny), np.int32(nz),
                grid=grid_size, block=block_size
            )

            return out_array
        
        return function

    def _prepare_workspace(self, dtype):
        out_shape = self._estimate_output_shape()

        # no need to resize the buffer
        if self._out_buffer is not None:
            if self._out_buffer.shape == out_shape:
                return

        self._out_buffer = np.empty(out_shape, dtype=dtype)

    def _estimate_output_shape(self):
        pass
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes
