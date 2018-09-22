import logging
from math import radians, sin, cos, ceil, hypot
import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from utoolbox.container import Resolution

logger = logging.getLogger(__name__)

class DeskewTransform(object):
    """
    TBA

    Note
    ----
    Shearing **always** happened along the X-axis.
    """
    def __init__(self, ctx, resolution, angle, rotate=True):
        """
        Parameters
        ----------
        ctx : pycuda.driver.Context
            PyCUDA context.
        resolution : Resolution
            TBA
        angle : float
            Angle between coverslip and detection objective.
        rotate : bool
            Rotate the result to perpendicular to coverslip, default to True.
        """
        ctx.push()

        if type(resolution) is not Resolution:
            try:
                resolution = Resolution._make(resolution)
            except TypeError:
                raise TypeError("invalid resolution input")
        angle = radians(angle)

        self._in_res = resolution
        self._angle = angle
        self._rotate = rotate

        self._px_shift = resolution.dz * cos(angle) / resolution.dxy
        self._out_res = Resolution(resolution.dxy, resolution.dz * sin(angle))
        logger.debug(
            "dyx={:.3f}um, dz={:.3f}um, angle={:.2f}rad, shift={:.2f}px".format(
                resolution.dxy, resolution.dz, angle, self.px_shift
            )
        )

        self._in_shape, self._out_shape = None, None
        self._upload_texture = None

        try:
            self._load_kernel()
        except cuda.CompileError:
            logger.error("compile error")
            raise

    def __call__(self, data, run_once=False):
        """
        Parameters
        ----------
        data : np.ndarray
            SPIM data.
        """
        if (data.shape != self._in_shape) or (data.dtype != self._dtype):
            logger.info("resizing workspace")
            if self._in_shape is not None:
                self.destroy_workspace()
            self._in_shape, self._dtype = data.shape, data.dtype
            self.create_workspace()

        # upload reference volume
        self._upload_texture(data)

        # execute
        # TODO determine grid/block size
        self._kernel.prepared_call(cuGrid, cuBlock, A_gpu.gpudata)
        # TODO transpose

        # copy back
        # TODO copy back

    @property
    def angle(self):
        return self._angle

    @property
    def px_shift(self):
        return self._px_shift

    @property
    def rotate(self):
        return self._rotate

#    @property
#    def spacing(self):
#        return self._spacing

    def create_workspace(self):
        nz, ny, nx = self._in_shape # [px]
        total_shift = self.px_shift * (nz-1) # [px]
        if self.rotate:
            dxy, dz = self._in_res # [um]
            rz, rx = nz*dz, nx*dxy # [um]
            # rotated dimension, resampled to be square voxel (almost)
            nw = ceil((rx * sin(self.angle)) / dxy)
            nv = ny
            h = hypot(total_shift*dxy, rz) # [um]
            nu = ceil((h + (rx * cos(self.angle))) / dxy)
            # update output resolution
            self._out_res = (dxy, dxy)
        else:
            nw, nv, nu = nz, ny, ceil(nx+total_shift)
        self._out_shape = (nv, nw, nu)

        self._buf = gpuarray.empty(self._out_shape, dtype=self._dtype, order='C')

    def destroy_workspace(self):
        pass

    def _load_kernel(self):
        path = os.path.join(os.path.dirname(__file__), "deskew.cu")
        with open(path, 'r') as fd:
            source = fd.read()
            module = SourceModule(source)

        self._kernel = module.get_function("deskew_kernel")
        self._texture = module.get_texref("ref_volume")

        self._kernel.prepare('P',texrefs=[self._texture])

    def _upload_texture(self, data):
        self.ref_volume = cuda.np_to_array(data, 'C', allowSurfaceBind=False)
        self._texture.set_array(self.ref_volume)








#    def __init__(self, cq, shift, rotate, resample):
#        self.context, self.queue = parse_cq(cq)
#
#        # load and compile
#        fpath = os.path.join(os.path.dirname(__file__), "deskew.cl")
#        with open(fpath, 'r') as fd:
#            source = fd.read()
#            program = cl.Program(self.context, source).build()
#            # select kernel to use
#            self.rotate = rotate
#            if rotate:
#                self.kernel = program.shear_and_rotate
#            else:
#                self.kernel = program.shear
#
#        # the uploaded raw data
#        self.ref_volume = None
#        # result host buffer
#        self.result = None
#        self.h_buf = None
#        self.d_buf = None
#
#        # pixels to shift
#        self.pixel_shift = shift
#
#    def __call__(self, volume, spacing):
#        # transpose zyx to yzx for 2D-layered texture
#        volume = volume.swapaxes(0, 1).copy()
#        self._upload_texture(volume)
#
#        # estimate dimension
#        dtype = volume.dtype
#        nw, nv0, nu0 = volume.shape
#        offset = ceil(self.pixel_shift * (nv0-1))
#        if self.rotate:
#            h = hypot(offset, nv0*spacing[1])
#            vsin = nv0/h * spacing[1]
#            vcos = offset/h
#
#            # rotated dimension and new origin
#            nu = ceil(nu0*vcos + h) / spacing[1]
#            nv = ceil(nu0*vsin) / spacing[1]
#
#            # offset
#            ov = ceil((nv0-nv)//2)
#        else:
#            nu = nu0 + offset
#            nv = nv0
#
#        # allocate host-side result buffer
#        if not (self.result and self.result.shape == (nw, nv, nu)):
#            coord = "{}x{}x{}".format(nv, nw, nu)
#            logger.info("output resized, {}".format(coord))
#
#            # result buffer
#            self.result = np.zeros(shape=(nw, nv, nu), dtype=dtype)
#
#            # staging buffers between host and device
#            self.h_buf = np.zeros(shape=(nv, nu), dtype=dtype)
#            self.d_buf = cl.Buffer(
#                self.context,
#                cl.mem_flags.WRITE_ONLY,
#                size=self.h_buf.nbytes
#            )
#
#            logger.info("... buffers reallocated")
#
#        p = 0
#        for iw in range(nw):
#            if self.rotate:
#                self.kernel.set_args(
#                    self.ref_volume,
#                    np.float32(vsin), np.float32(vcos),
#                    np.float32(self.pixel_shift),
#                    np.int32(iw),
#                    np.int32(nu), np.int32(nv),
#                    np.int32(ov),
#                    self.d_buf
#                )
#            else:
#                self.kernel.set_args(
#                    self.ref_volume,
#                    np.float32(self.pixel_shift),
#                    np.int32(iw),
#                    np.int32(nu), np.int32(nv),
#                    self.d_buf
#                )
#            cl.enqueue_nd_range_kernel(
#                self.queue,
#                self.kernel,
#                self.h_buf.shape,   # global size
#                None                # local size
#            )
#            cl.enqueue_copy(self.queue, self.h_buf, self.d_buf)
#            self.result[iw, ...] = self.h_buf
#
#            pp = int((iw+1)/nw * 10)
#            if pp > p:
#                p = pp
#                logger.debug("{}%".format(p*10))
#
#        # transpose back
#        return self.result.swapaxes(0, 1).copy()
#
#    def _upload_texture(self, array):
#        # force convert to float for linear interpolation
#        array = array.astype(np.float32).copy()
#
#        dtype = array.dtype
#        shape = array.shape
#        strides = array.strides
#
#        self.ref_volume = cl.Image(
#            self.context,
#            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#            cl.ImageFormat(
#                cl.channel_order.R,
#                cl.DTYPE_TO_CHANNEL_TYPE[dtype]
#            ),
#            shape=shape[::-1],
#            pitches=strides[::-1][1:],
#            hostbuf=array,
#            is_array=True
#        )
