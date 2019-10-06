import logging
from math import ceil
import os

import coloredlogs
import cupy as cp
import imageio
from mako.template import Template
import numpy as np

from utoolbox.container.datastore import ImageFolderDatastore
from utoolbox.stitching.fusion import rmlp2
from utoolbox.utils.decorator import timeit

##
# region: Setup logger
##
logging.getLogger("tifffile").setLevel(logging.ERROR)

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
##
# endregion
##

##
# region: Load data
##
# I = imageio.imread('lena512.bmp')
# I = np.arange(0, 127*127, dtype=np.uint16).reshape((127, 127))
# I = I.astype(np.float32)
# print(I)
# imageio.imwrite("I.tif", I)

ds = ImageFolderDatastore(
    #'/Users/Andy/Documents/utoolbox/data/fusion/crop',
    #'/Users/Andy/Documents/Sinica (Data)/Projects/ExM SIM/20181224_Expan_Tub_Tiling_SIM',
    "field_test_data",
    imageio.imread,
    # pattern='RAWcell1_*'
)

Iz = []
iz = 0
for fn, I in ds.items():
    print('reading "{}"'.format(fn))
    Iz.append(I.astype(np.float32))
print(Iz)
##
# endregion
##

##
# region: Kernel definitions
##

TILE_WIDTH = 16

cu_file = os.path.join(os.path.dirname(__file__), "rmlp2.cu")

with open(cu_file, "r") as fd:
    template = Template(fd.read())
    source = template.render(tile_width=TILE_WIDTH)

modified_laplacian_kernel = cp.RawKernel(source, "modified_laplacian_kernel")
sml_kernel = cp.RawKernel(source, "sml_kernel")
keep_max_kernel = cp.RawKernel(source, "keep_max_kernel")

###
# endregion
###


def cpu_modified_laplacian(I):
    J = np.empty_like(I)
    n, m = I.shape
    for y in range(0, n):
        for x in range(0, m):
            pc = I[y, x]
            # symmetric padding, pad size 1
            pu = I[y + 1, x] if y < n - 1 else I[y - 1, x]
            pd = I[y - 1, x] if y > 0 else I[y + 1, x]
            pr = I[y, x + 1] if x < m - 1 else I[y, x - 1]
            pl = I[y, x - 1] if x > 0 else I[y, x + 1]
            J[y, x] = abs(2 * pc - pl - pr) + abs(2 * pc - pu - pd)
    return J


def cpu_sml(I, T=1 / 255.0):
    G = cpu_modified_laplacian(I)
    Gp = np.zeros((3, 3))  # buffer area
    S = np.empty_like(I)
    n, m = S.shape
    for y in range(0, n):
        for x in range(0, m):
            pu = min(y + 1, n - 1)
            pd = max(y - 1, 0)
            pr = min(x + 1, m - 1)
            pl = max(x - 1, 0)
            Gp = G[pd : pu + 1, pl : pr + 1]
            S[y, x] = np.sum(Gp[Gp >= T])
    return S


@timeit
def cpu_generate_init_mask(I, T):
    """
    Generate mask estimation based on SML.

    Parameters
    ----------
    I : list of np.ndarray
        List of original raw images.
    T : float
        Blur level criteria.
    """
    M = np.full(I[0].shape, 0, dtype=np.uint32)
    V = np.full_like(I[0], np.NINF)
    n, m = I[0].shape
    for i, iI in enumerate(I):
        S = cpu_sml(iI, T)
        M[abs(S) > V] = i + 1
        V[abs(S) > V] = S[abs(S) > V]

    return M


def gpu_modified_laplacian(I):
    ny, nx = I.shape

    J = cp.empty_like(I)
    block_sz = (TILE_WIDTH, TILE_WIDTH)
    grid_sz = (int(ceil(nx / TILE_WIDTH)), int(ceil(ny / TILE_WIDTH)))
    modified_laplacian_kernel(grid_sz, block_sz, (J, I, nx, ny))

    return J


def gpu_sml(I, T=1 / 255.0):
    ny, nx = I.shape

    J = cp.empty_like(I)

    block_sz = (TILE_WIDTH, TILE_WIDTH)
    grid_sz = (int(ceil(nx / TILE_WIDTH)), int(ceil(ny / TILE_WIDTH)))

    modified_laplacian_kernel(grid_sz, block_sz, (J, I, nx, ny))

    sml_kernel(grid_sz, block_sz, (J, J, nx, ny, T))

    return J


@timeit
def gpu_generate_init_mask(I, T):
    S = cp.array(I[0])
    nelem = S.size

    n_threads = 1024
    block_sz = (n_threads,)
    grid_sz = (int(ceil(nelem / n_threads)),)

    V = gpu_sml(S, T)
    M = cp.ones_like(V, dtype=cp.int32)
    for i, iI in enumerate(I[1:], 2):
        S = cp.array(iI)
        S = gpu_sml(S, T)
        keep_max_kernel(grid_sz, block_sz, (M, V, S, nelem, i))
    return cp.asnumpy(M)


if False:
    print("==gpu==")
    J_gpu = gpu_generate_init_mask(Iz, 1 / 255.0)
    print(J_gpu)
    imageio.imwrite("J_gpu.tif", cp.asnumpy(J_gpu))

    print("==cpu==")
    J_cpu = cpu_generate_init_mask(Iz, 1 / 255.0)
    print(J_cpu)
    imageio.imwrite("J_cpu.tif", J_cpu)
else:
    print("==rmlp==")
    Rz = rmlp2(Iz, T=1 / 255.0, r=4, K=12)
    imageio.imwrite("rmlp_test.tif", Rz)
