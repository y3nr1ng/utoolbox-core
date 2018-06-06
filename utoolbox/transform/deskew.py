import os
import itertools
import logging
logger = logging.getLogger(__name__)
import math

import numpy as np
import pyopencl as cl

from utoolbox.container import Raster
from utoolbox.container.layouts import Volume
from utoolbox.utils.files import convert_size

class DeskewTransform(object):
    def __init__(self, shift):
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
        logger.debug(self.device.get_info(cl.device_info.NAME))
        self.context = cl.Context(
            devices=[self.device],
            properties=[(cl.context_properties.PLATFORM, _platform)]
        )
        self.queue = cl.CommandQueue(self.context)
        fpath = os.path.join(os.path.dirname(__file__), "deskew.cl")
        with open(fpath, 'r') as fd:
            source = fd.read()
            self.program = cl.Program(self.context, source).build(devices=[self.device])

        # the uploaded raw data
        self.ref_vol = None

        # pixels to shift
        self.pixel_shift = shift

    def __call__(self, volume):
        # transpose zyx to yzx for 2D-layered texture
        volume = volume.swapaxes(0, 1).copy()
        self._upload_texture(volume)

        # allocate host-side result buffer
        dtype = volume.dtype
        nw, nv, nu = volume.shape
        nu += int(math.ceil(self.pixel_shift * (nv-1)))
        result = np.zeros(shape=(nw, nv, nu), dtype=dtype)

        # layer buffer
        h_buf = np.zeros(shape=(nv, nu), dtype=dtype)
        d_buf = cl.Buffer(
            self.context,
            cl.mem_flags.WRITE_ONLY,
            size=h_buf.nbytes
        )

        kernel = self.program.shear
        p = 0
        for iw in range(nw):
            kernel.set_args(
                self.ref_vol,
                np.float32(self.pixel_shift),
                np.int32(iw),
                np.int32(nu), np.int32(nv),
                d_buf
            )
            cl.enqueue_nd_range_kernel(self.queue, kernel, h_buf.shape, None)
            cl.enqueue_copy(self.queue, h_buf, d_buf)
            result[iw, ...] = h_buf

            pp = int((iw+1)/nw * 10)
            if pp > p:
                p = pp
                logger.debug("{}%".format(p*10))

        d_buf.release()

        # transpose back
        return result.swapaxes(0, 1).copy()

    def _upload_texture(self, array):
        # force convert to float for linear interpolation
        array = array.astype(np.float32).copy()

        dtype = array.dtype
        shape = array.shape
        strides = array.strides

        self.ref_vol = cl.Image(
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

def deskew(volume, shift):
    """Deskew acquired SPIM volume of specified angle.

    Parameters
    ----------
    volume : Raster
        SPIM data.
    shift : float
        Sample stage shift range, in um.

    Note
    ----
    - During the resampling process, inhomogeneous spacing will also take into
      account and normalized.
    - Shearing **always** happened along the X-axis.
    """
    try:
        spacing = volume.metadata.spacing
    except AttributeError:
        spacing = (1, ) * volume.ndim
    dtype = volume.dtype

    if not issubclass(dtype.type, np.integer):
        raise TypeError("not an integer raster")

    pixel_shift = shift / spacing[2]
    transform = DeskewTransform(pixel_shift)
    result = transform(volume)

    #result = result.astype(dtype)

    return result
