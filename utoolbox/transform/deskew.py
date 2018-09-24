import logging
from math import radians, sin, cos, ceil, hypot
import os

from jinja2 import Template
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.tools import dtype_to_ctype
import tqdm

from utoolbox.container import AttrDict, Resolution

logger = logging.getLogger(__name__)

class DeskewTransform(object):
    """
    TBA

    Note
    ----
    Shearing **always** happened along the X-axis.
    """
    def __init__(self, resolution, angle, rotate=True):
        """
        Parameters
        ----------
        resolution : Resolution
            TBA
        angle : float
            Angle between coverslip and detection objective.
        rotate : bool
            Rotate the result to perpendicular to coverslip, default to True.
        """
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
        self._iw_origin = 0
        self._upload_texture = None

    def __call__(self, data, run_once=False, copy=False):
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

            try:
                self._load_kernel()
            except cuda.CompileError:
                logger.error("compile error")
                raise

            cuda.memcpy_htod(self._d_px_shift, np.float32(self.px_shift))

            self.create_workspace()

        # upload reference volume
        ref_vol = self._upload_ref_vol(data)

        # determine grid & block size
        n_blocks = (32, 32, 1)
        nw, nv, nu = self._out_shape
        n_grids = (ceil(float(nu)/n_blocks[0]), ceil(float(nw)/n_blocks[1]), 1)

        nz, _, nx = self._in_shape
        dxy, dz = self._in_res
        duv, dw = self._out_res

        # execute
        for iv in tqdm.trange(nv):
            self._kernel.prepared_call(
                n_grids, n_blocks,
                self._d_slice.gpudata,
                np.int32(iv),
                np.int32(nu), np.int32(nw),
                np.int32(nx), np.int32(nz)
            )
            h_slice = self._d_slice.get()
            self._h_vol[:, iv, :] = h_slice

        # free resource
        ref_vol.free()

        result = self._h_vol if not copy else self._h_vol.copy()
        return result

    @property
    def angle(self):
        return self._angle

    @property
    def out_res(self):
        return self._out_res

    @property
    def px_shift(self):
        return self._px_shift

    @property
    def rotate(self):
        return self._rotate

    def create_workspace(self):
        nz, ny, nx = self._in_shape # [px]
        total_shift = self.px_shift * (nz-1) # [px]

        if self.rotate:
            h = hypot(total_shift, nz) # [px]
            vsin, vcos = nz/h, total_shift/h # from [px]

            nw, nv, nu = ceil(nx*vsin), ny, ceil(h + nx*vcos)

            # update output resolution
            dxy, dz = self._in_res # [um]
            self._out_res = Resolution(
                hypot(total_shift*dxy, nz*dz) / nu,
                nx*dxy / nw
            )
            #TODO need to findout whether resample is required
        else:
            vsin, vcos = 0., 1.
            nw, nv, nu = nz, ny, ceil(nx+total_shift)
        self._out_shape = (nw, nv, nu)
        logger.debug("duv={:.3f}um, dw={:.3f}um".format(self._out_res.dxy, self._out_res.dz))

        cuda.memcpy_htod(self._d_vsin, np.float32(vsin))
        cuda.memcpy_htod(self._d_vcos, np.float32(vcos))
        logger.debug("vsin={:.4f}, vcos={:.4f}".format(vsin, vcos))

        self._h_vol = np.empty(self._out_shape, dtype=self._dtype)
#        self._h_vol = np.empty(self._out_shape, dtype=np.float32)
        self._d_slice = gpuarray.empty((nw, nu), dtype=self._dtype, order='C')
#        self._d_slice = gpuarray.empty((nw, nu), dtype=np.float32, order='C')
        logger.info("workspace allocated, {}, {}".format(self._out_shape, self._dtype))

    def destroy_workspace(self):
        pass

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
        ref_vol = cuda.np_to_array(data.astype(np.float32), 'C')
        self._texture.set_array(ref_vol)
        return ref_vol








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
