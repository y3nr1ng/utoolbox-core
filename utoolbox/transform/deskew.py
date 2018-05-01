import logging
logger = logging.getLogger(__name__)
import math

import numpy as np
import pycuda.autoinit
from pycuda import compiler, driver, gpuarray

from utoolbox.container import Raster
from utoolbox.container.layouts import Volume

_shear_kernel_source = """
texture<int, cudaTextureType2DLayered, cudaReadModeElementType> tex;

__global__
void shear_kernel(
    const float factor,
    int iv,
    const int nu, const int nw, // output size
    float *result
) {
    const int iu = blockIdx.x*blockDim.x + threadIdx.x;
    const int iw = blockIdx.y*blockDim.y + threadIdx.y;
    if ((iu >= nu) || (iw >= nw)) {
        return;
    }

    const int i = iv * (nu*nw) + iw * nu + iu;

    float ix = (iu - factor*iw) + 0.5f;
    float iz = iw + 0.5f;

    result[i] = tex2DLayered(tex, ix, iz, iv);
}
"""
_shear_kernel_module = compiler.SourceModule(_shear_kernel_source)
_shear_kernel_function = _shear_kernel_module.get_function("shear_kernel")
_shear_src_texref = _shear_kernel_module.get_texref("tex")

def _shear_subblock(volume, origin, shape, spacing, offset, blocks=(16, 16, 1)):
    pass

def _estimate_deskew_parameters(shape, spacing, angle):
    angle = math.radians(angle)
    dz, dy, dx = spacing
    pixel_offset = dz / math.tan(angle) / dx

    nz, ny, nx = shape
    nx += int(math.ceil(pixel_offset * (nz-1)))

    return (nz, ny, nx), pixel_offset

def _shear(volume, spacing, angle, blocks=(16, 16, 1)):
    shape = volume.shape
    nz, ny, nx = shape
    dz, dy, dz = spacing

    shape, offset = _estimate_deskew_parameters(shape, spacing, angle)
    nw, nv, nu = shape
    logger.info("shape {} -> {}".format(volume.shape, shape))
    logger.info("offset(px)={:.5f}".format(offset))

    # - only signed data type is supported as texel
    #   (CUDA Programming Guide, 3.2.11.1, (3))
    # - swap dimension (ZYX -> YZX) for better caching on GPU
    volume = volume.astype(np.int32).swapaxes(0, 1).copy()
    logger.debug("before")
    logger.debug(".. shape={}".format(volume.shape))
    logger.debug(".. min={}".format(volume.min()))
    logger.debug(".. max={}".format(volume.max()))

    # create array descriptor for texture binding
    desc = driver.ArrayDescriptor3D()
    desc.width = nx
    desc.height = nz
    desc.depth = ny
    desc.format = driver.array_format.SIGNED_INT32
    desc.num_channels = 1
    desc.flags |= driver.array3d_flags.ARRAY3D_LAYERED #TODO patch upstream

    # upload to array
    a_volume = driver.Array(desc)
    copy_func = driver.Memcpy3D()
    copy_func.set_src_host(volume)
    copy_func.set_dst_array(a_volume)
    copy_func.width_in_bytes = copy_func.src_pitch = volume.strides[1]
    copy_func.src_height = copy_func.height = nz
    copy_func.depth = ny
    copy_func()

    # bind array to texture
    _shear_src_texref.set_array(a_volume)
    _shear_src_texref.set_address_mode(0, driver.address_mode.BORDER)
    _shear_src_texref.set_address_mode(1, driver.address_mode.BORDER)
    # returned data type has to be float
    # (CUDA Programming Guide, 3.2.11.1, (7))
    _shear_src_texref.set_filter_mode(driver.filter_mode.LINEAR)

    result = np.zeros(shape=(nv, nw, nu), dtype=np.float32)

    grids = (-(-nu//blocks[0]), -(-nw//blocks[1]), 1)
    for iv in range(nv):
        _shear_kernel_function(
            np.float32(offset),
            np.int32(iv),
            np.int32(nu), np.int32(nw),
            driver.Out(result),
            grid=grids, block=blocks, texrefs=[_shear_src_texref]
        )

    # free the resource
    a_volume.free()

    logger.debug("after")
    logger.debug(".. min={}".format(result.min()))
    logger.debug(".. max={}".format(result.max()))

    result = result.swapaxes(0, 1)
    return result

def deskew(volume, angle, resample=False):
    """Deskew acquired SPIM volume of specified angle.

    The deskew process can be treated as two steps: shear and rotate. Data
    generated through sample scan method will contain an implicit angle due to
    the orientation of the sample holder, shearing restore the volume to its
    intended spatial topology. To display the layered data without elevation
    angle from the surface, rotation is required, however, this resampling
    process will cause voxel intensity deviate from their true value further.

    Parameters
    ----------
    volume : Raster
        SPIM data.
    angle : float
        Elevation of the objective lens with respect to the sample holder in
        degrees following the right-hand rule.
    resample : bool, default to False
        If true, rotate the volume to global coordinate.

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

    result = _shear(volume, spacing, angle)
    if resample:
        raise NotImplementedError
    #result = result.astype(dtype)

    return result
