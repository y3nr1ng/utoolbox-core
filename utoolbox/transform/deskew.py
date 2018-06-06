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
                    if device.get_info(cl.device_info.VENDOR_ID) == 16918272:
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
        nw, nv, nu = volume.shape
        nu += int(math.ceil(self.pixel_shift * (nv-1)))
        result = np.zeros(shape=(nw, nv, nu), dtype=volume.dtype)

#        # determine block size from remaining spaces, 200MB
#        factor = self._calculate_split_factor(result, 200*2**20)
#        offset, bs = self._generate_block_info(result.shape, factor)
#        logger.debug("block size = {}".format(bs))

        # layer buffer
        d_buf = cl.Buffer(
            self.context,
            cl.mem_flags.WRITE_ONLY,
            size=volume[0, ...].nbytes
        )

        kernel = self.program.shear
        for iw in range(nw):
            logger.debug(".. kernel")
            kernel.set_args(
                self.ref_vol,
                np.float32(self.pixel_shift),
                np.int32(iw),
                np.int32(nu), np.int32(nv),
                d_buf
            )
            cl.enqueue_nd_range_kernel(
                self.queue,
                kernel,
                (nu, nv),   # global size
                None       # local size
            )
            logger.debug(".. copy")
            cl.enqueue_copy(self.queue, result[iw, ...], d_buf)
            logger.debug("{} / {}".format(iw+1, nw))

        d_buf.release()

        # transpose back
        return result.copy()

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

    def _calculate_split_factor(self, array, mem_lim):
        """
        Calculate how the block size should split in order to fit in device memory.
        """
        nw, nv, nu = array.shape
        bytes_required = (nu*nv*nw) * np.dtype(array.dtype).itemsize
        logger.debug("requires {} / available {}".format(
            convert_size(bytes_required), convert_size(mem_lim))
        )

        fu = 1
        fv = 1
        # divide the largest dimension (Y or Z) in half and repeat
        while bytes_required > mem_lim:
            if nu >= nv:
                fu *= 2
                nu //= 2
            else:
                fv *= 2
                nv //= 2
            bytes_required /= 2

        factor = (1, fv, fu)
        logger.debug("split factor = {}".format(factor))

        return factor

    def _generate_block_info(self, shape, factor):
        nw, nv, nu = shape
        fw, fv, fu = factor

        # block size
        bu = -(-nu//fu)
        bv = -(-nv//fv)
        bw = -(-nw//fw)

        # location of the offsets
        ou = list(range(0, nu, bu))
        ov = list(range(0, nv, bv))
        ow = list(range(0, nw, bw))

        return list(itertools.product(ow, ov, ou)), (bw, bv, bu)

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
