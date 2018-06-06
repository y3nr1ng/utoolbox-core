import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pyopencl as cl

from utoolbox.container import Raster
from utoolbox.container.layouts import Volume

class DeskewTransform(object):
    def __init__(self, shift, rotate, resample):
        #TODO auto prioritize
        platform = None
        try:
            for _platform in cl.get_platforms():
                for device in _platform.get_devices(device_type=cl.device_type.GPU):
                    #if device.get_info(cl.device_info.VENDOR_ID) == 16918272:
                    platform = _platform
                    self.device = device
                    raise StopIteration
        except:
            pass
        logger.debug("Using '{}'".format(self.device.get_info(cl.device_info.NAME)))
        self.context = cl.Context(
            devices=[self.device],
            properties=[(cl.context_properties.PLATFORM, _platform)]
        )
        self.queue = cl.CommandQueue(self.context)

        # load and compile
        fpath = os.path.join(os.path.dirname(__file__), "deskew.cl")
        with open(fpath, 'r') as fd:
            source = fd.read()
            program = cl.Program(self.context, source).build()
            # select kernel to use
            if rotate:
                self.kernel = program.shear_and_rotate
            else:
                self.kernel = program.shear

        # the uploaded raw data
        self.ref_volume = None
        # result host buffer
        self.result = None

        # pixels to shift
        self.pixel_shift = shift

    def __call__(self, volume):
        # transpose zyx to yzx for 2D-layered texture
        volume = volume.swapaxes(0, 1).copy()
        self._upload_texture(volume)

        # allocate host-side result buffer
        dtype = volume.dtype
        nw, nv, nu = volume.shape
        nu += int(-(-(self.pixel_shift * (nv-1))//1))
        if not (self.result and self.result.shape == (nw, nv, nu)):
            self.result = np.zeros(shape=(nw, nv, nu), dtype=dtype)

        # layer buffer
        h_buf = np.zeros(shape=(nv, nu), dtype=dtype)
        d_buf = cl.Buffer(
            self.context,
            cl.mem_flags.WRITE_ONLY,
            size=h_buf.nbytes
        )

        p = 0
        for iw in range(nw):
            self.kernel.set_args(
                self.ref_volume,
                np.float32(self.pixel_shift),
                np.int32(iw),
                np.int32(nu), np.int32(nv),
                d_buf
            )
            cl.enqueue_nd_range_kernel(self.queue, self.kernel, h_buf.shape, None)
            cl.enqueue_copy(self.queue, h_buf, d_buf)
            self.result[iw, ...] = h_buf

            pp = int((iw+1)/nw * 10)
            if pp > p:
                p = pp
                logger.debug("{}%".format(p*10))

        d_buf.release()

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

def deskew(data, shift, rotate=False, resample=False):
    """Deskew acquired SPIM volume of specified angle.

    Parameters
    ----------
    volume : Raster
        SPIM data.
    shift : float
        Sample stage shift range, in um.
    angle : bool, default to False
        True to rotate the result to perpendicular to coverslip.

    Note
    ----
    Shearing **always** happened along the X-axis.
    """
    try:
        spacing = data.metadata.spacing
    except AttributeError:
        spacing = (1, ) * data.ndim

    if not np.issubdtype(data.dtype, np.uint16):
        raise TypeError("only 16-bit unsigned integer is supported")

    pixel_shift = shift / spacing[2]
    transform = DeskewTransform(pixel_shift, rotate, resample)
    result = transform(data)

    return result
