import logging
from math import ceil, cos, radians, sin
import os

import cupy as cp
from cupy.cuda import runtime
from cupy.cuda.texture import (
    ChannelFormatDescriptor,
    CUDAarray,
    ResourceDescriptor,
    TextureDescriptor,
    TextureObject,
)
import numpy as np

from utoolbox.parallel import RawKernelFile

__all__ = ["deskew"]

logger = logging.getLogger(__name__)

cu_file = os.path.join(os.path.dirname(__file__), "deskew.cu")
kernels = RawKernelFile(cu_file)


def deskew(data, angle, dx, dz, rotate=True, return_resolution=True, out=None):
    """
    Args:
        data (ndarray): 3-D array to apply deskew
        angle (float): angle between the objective and coverslip, in degree
        dx (float): X resolution
        dz (float): Z resolution
        rotate (bool, optional): rotate and crop the output
        return_resolution (bool, optional): return deskewed X/Z resolution
        out (ndarray, optional): array to store the result
    """
    angle = radians(angle)

    # shift along X axis, in pixels
    shift = dz * cos(angle) / dx
    logger.debug(f"layer shift: {shift:.04f} px")

    # estimate new size
    nw, nv, nu = data.shape
    nz, ny, nx = nw, nv, nu + ceil(shift * (nw - 1))

    # upload texture
    ch = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    arr = CUDAarray(ch, nu, nw)
    res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)

    address_mode = (runtime.cudaAddressModeBorder, runtime.cudaAddressModeBorder)
    tex = TextureDescriptor(
        address_mode, runtime.cudaFilterModeLinear, runtime.cudaReadModeElementType
    )

    # transpose
    data = np.swapaxes(data, 0, 1)
    data = np.ascontiguousarray(data)

    data_in = data.astype(np.float32)
    data_out = cp.empty((ny, nz, nx), np.float32)
    for i, layer in enumerate(data_in):
        arr.copy_from(layer)  # TODO use stream
        texobj = TextureObject(res, tex)

        kernels["shear_kernel"](
            (ceil(nx / 16), ceil(nz / 16)),
            (16, 16),
            (data_out[i, ...], texobj, nx, nz, nu, np.float32(shift)),
        )

    data_out = cp.swapaxes(data_out, 0, 1)
    data_out = cp.asnumpy(data_out)
    data_out = data_out.astype(data.dtype)

    if return_resolution:
        # new resolution
        dz *= sin(angle)
        return data_out, (dz, dx)
    else:
        return data_out


if __name__ == "__main__":
    import coloredlogs
    import imageio

    logging.getLogger("tifffile").setLevel(logging.ERROR)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    data = imageio.volread(
        "cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs.tif"
    )
    print(data.shape)
    result, res = deskew(data, 32.8, 0.103, 0.5, rotate=True)
    print(res)
    print(result.shape)
    imageio.volwrite("result.tif", result)

