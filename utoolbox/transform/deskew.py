import math
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit

import numpy as np

from utoolbox.container import Volume

_shear_kernel_source = """
texture<float, cudaTextureType2DLayered> tex;

__global__
void shear_kernel(
    const float factor,
    const int iz,
    const int nx, const int ny,   // input size
    const int nu, const int nv,   // output size
    float *result
) {
    const int iu = blockIdx.x * blockDim.x + threadIdx.x;
    const int iv = blockIdx.y * blockDim.y + threadIdx.y;
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    // calculate linear memory index
    const int i = (iz * (nu*nv)) + (iv*nu) + iu;

    // calculate the coordinate before transformation
    const float ix = iu - factor*iz;
    const float iy = iv;

    result[i] = tex2DLayered(tex, ix, iy, iz);
}
"""
_shear_kernel_module = pycuda.compiler.SourceModule(_shear_kernel_source)
_shear_kernel_function = _shear_kernel_module.get_function("shear_kernel")
_shear_src_texref = _shear_kernel_module.get_texref("tex")

def _shear(volume, angle, interpolation='linear', blocks=(16, 16, 1)):
    angle = math.radians(angle)
    dz, dy, dx = volume.resolution
    pixel_offset = dz * math.cos(angle) / dx
    print("offset={}".format(pixel_offset))

    nz, ny, nx = volume.shape
    nx = int(math.ceil(dx + pixel_offset * (nz-1)))
    print("new_size=({}, {}, {})".format(nz, ny, nx))

    cuda.matrix_to_texref(volume, _shear_src_texref, order='C')
    _shear_src_texref.set_filter_mode(cuda.filter_mode.LINEAR)

    grids = (nx // blocks[0], ny // blocks[1], 1)
    print(grids)

    result = np.zeros_like(volume)
    for iz in range(nz):
        _shear_kernel_function(
            np.float32(pixel_offset),
            iz,
            volume.shape[1], volume.shape[0],
            nx, ny,
            cuda.Out(result),
            texrefs=[_shear_src_texref], blocks=bocks, grid=grids
        )
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
    volume : Volume
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
    if not isinstance(volume, Volume):
        raise ValueError("utoolbox.container.Volume is required.")
    dtype = volume.dtype

    volume = volume.astype(np.float32)
    result = _shear(volume, angle)
    if resample:
        raise NotImplementedError
    result = result.astype(dtype)

    return result
