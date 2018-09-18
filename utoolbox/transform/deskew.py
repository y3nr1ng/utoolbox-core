import logging
from math import radians, cos, ceil, hypot
import os

import numpy as np
import pyopencl as cl

from utoolbox.container import Resolution
from utoolbox.parallel import parse_cq

logger = logging.getLogger(__name__)

class DeskewTransform(object):
    """
    TBA

    Note
    ----
    Shearing **always** happened along the X-axis.
    """
    def __init__(self, cq, resolution, angle, rotate=True):
        """
        Parameters
        ----------
        cq : TBA
            OpenCL context or command queue.
        resolution : Resolution
            TBA
        angle : float
            Angle between coverslip and detection objective.
        rotate : bool
            Rotate the result to perpendicular to coverslip, default to True.
        """
        self.context, self.queue = parse_cq(cq)

        if type(resolution) is not Resolution:
            try:
                resolution = Resolution._make(resolution)
            except TypeError:
                raise TypeError("invalid resolution input")
        angle = radians(angle)

        self._in_res = resolution
        self._angle = angle
        self.rotate = rotate

        self._px_shift = resolution.dz * cos(angle)
        self._out_res = Resolution(resolution.dxy, resolution.dz * sin(angle))
        logger.debug(
            "dyx={:.4f}um, dz={:.4f}um, angle={:2.f}rad, shift={:.4f}px".format(
                resolution.dxy, resolution.dz, angle, self.px_shift
            )
        )

        self._load_kernel()

    def __enter__(self):
        self.create_workspace()
        return self

    def __exit__(self, *args):
        self.destroy_workspace()

    def __call__(self, data):
        """
        Parameters
        ----------
        data : np.ndarray
            SPIM data.
        """
        if not np.issubdtype(data.dtype, np.uint16):
            raise TypeError("only 16-bit unsigned integer is supported")
        pass
    
    @property
    def angle(self):
        return _angle

    @property
    def px_shift(self):
        return self._px_shift

    @property
    def rotate(self):
        return self._rotate

    @property
    def spacing(self):
        return self._spacing

    def create_workspace(self):
        pass

    def destroy_workspace(self):
        pass

    def _load_kernel(self):
        fpath = os.path.join(os.path.dirname(__file__), "deskew.cl")
        with open(fpath, 'r') as fd:
            source = fd.read()
            program = cl.Program(self.context, source).build()
            #TODO load kerenl



    def __init__(self, cq, shift, rotate, resample):
        self.context, self.queue = parse_cq(cq)

        # load and compile
        fpath = os.path.join(os.path.dirname(__file__), "deskew.cl")
        with open(fpath, 'r') as fd:
            source = fd.read()
            program = cl.Program(self.context, source).build()
            # select kernel to use
            self.rotate = rotate
            if rotate:
                self.kernel = program.shear_and_rotate
            else:
                self.kernel = program.shear

        # the uploaded raw data
        self.ref_volume = None
        # result host buffer
        self.result = None
        self.h_buf = None
        self.d_buf = None

        # pixels to shift
        self.pixel_shift = shift

    def __call__(self, volume, spacing):
        # transpose zyx to yzx for 2D-layered texture
        volume = volume.swapaxes(0, 1).copy()
        self._upload_texture(volume)

        # estimate dimension
        dtype = volume.dtype
        nw, nv0, nu0 = volume.shape
        offset = ceil(self.pixel_shift * (nv0-1))
        if self.rotate:
            h = hypot(offset, nv0*spacing[1])
            vsin = nv0/h * spacing[1]
            vcos = offset/h

            # rotated dimension and new origin
            nu = ceil(nu0*vcos + h) / spacing[1]
            nv = ceil(nu0*vsin) / spacing[1]

            # offset
            ov = ceil((nv0-nv)//2)
        else:
            nu = nu0 + offset
            nv = nv0

        # allocate host-side result buffer
        if not (self.result and self.result.shape == (nw, nv, nu)):
            coord = "{}x{}x{}".format(nv, nw, nu)
            logger.info("output resized, {}".format(coord))

            # result buffer
            self.result = np.zeros(shape=(nw, nv, nu), dtype=dtype)

            # staging buffers between host and device
            self.h_buf = np.zeros(shape=(nv, nu), dtype=dtype)
            self.d_buf = cl.Buffer(
                self.context,
                cl.mem_flags.WRITE_ONLY,
                size=self.h_buf.nbytes
            )

            logger.info("... buffers reallocated")

        p = 0
        for iw in range(nw):
            if self.rotate:
                self.kernel.set_args(
                    self.ref_volume,
                    np.float32(vsin), np.float32(vcos),
                    np.float32(self.pixel_shift),
                    np.int32(iw),
                    np.int32(nu), np.int32(nv),
                    np.int32(ov),
                    self.d_buf
                )
            else:
                self.kernel.set_args(
                    self.ref_volume,
                    np.float32(self.pixel_shift),
                    np.int32(iw),
                    np.int32(nu), np.int32(nv),
                    self.d_buf
                )
            cl.enqueue_nd_range_kernel(
                self.queue,
                self.kernel,
                self.h_buf.shape,   # global size
                None                # local size
            )
            cl.enqueue_copy(self.queue, self.h_buf, self.d_buf)
            self.result[iw, ...] = self.h_buf

            pp = int((iw+1)/nw * 10)
            if pp > p:
                p = pp
                logger.debug("{}%".format(p*10))

        # transpose back
        return self.result.swapaxes(0, 1).copy()

    def _upload_texture(self, array):
        # force convert to float for linear interpolation
        array = array.astype(np.float32).copy()

        dtype = array.dtype
        shape = array.shape
        strides = array.strides

        self.ref_volume = cl.Image(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            cl.ImageFormat(
                cl.channel_order.R,
                cl.DTYPE_TO_CHANNEL_TYPE[dtype]
            ),
            shape=shape[::-1],
            pitches=strides[::-1][1:],
            hostbuf=array,
            is_array=True
        )
