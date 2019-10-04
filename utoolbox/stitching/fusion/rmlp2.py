import logging
from math import ceil
import os

import cupy as cp
import imageio
from mako.template import Template
from numba import jit
import numpy as np
from skimage.transform import resize
from scipy import ndimage as ndi

from utoolbox.utils.decorator import timeit

__all__ = ["rmlp2"]

logger = logging.getLogger(__name__)


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


def _smooth(image, sigma, mode, cval):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty(image.shape, dtype=np.float32)

    # apply Gaussian filter to all channels independently
    ndi.gaussian_filter(image, sigma, output=smoothed, mode=mode, cval=cval)
    return smoothed


def _pyramid_laplacian(
    image, max_layer=-1, downscale=2, sigma=None, order=1, mode="reflect", cval=0
):
    """Yield images of the laplacian pyramid formed by the input image.
    Each layer contains the difference between the downsampled and the
    downsampled, smoothed image::
        layer = resize(prev_layer) - smooth(resize(prev_layer))
    Note that the first image of the pyramid will be the difference between the
    original, unscaled image and its smoothed version. The total number of
    images is `max_layer + 1`. In case all layers are computed, the last image
    is either a one-pixel image or the image where the reduction does not
    change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf
    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html
    """
    # multichannel = _multichannel_default(multichannel, image.ndim)
    # _check_factor(downscale)
    assert downscale > 1

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    layer = 0
    current_shape = image.shape
    out_shape = tuple([ceil(d / float(downscale)) for d in current_shape])

    smoothed_image = _smooth(image, sigma, mode, cval)
    yield image - smoothed_image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        resized_image = resize(
            smoothed_image,
            out_shape,
            order=order,
            mode=mode,
            cval=cval,
            anti_aliasing=False,
        )
        smoothed_image = _smooth(resized_image, sigma, mode, cval)

        current_shape = np.asarray(resized_image.shape)
        out_shape = tuple([ceil(d / float(downscale)) for d in current_shape])

        last_layer = np.all(current_shape == out_shape) or layer == max_layer - 1
        if last_layer:
            yield resized_image
            break
        else:
            yield resized_image - smoothed_image


def pyramid_fusion(images, M, K, sigma=None):
    """
    Fused pyramid layers using the mask.

    Parameters
    ----------
    I : np.ndarray
        The source image.
    M : np.ndarray
        The masked image.
    K : int
        Level of the pyramid.
    """
    # automatically determine sigma which covers > 99% of distribution
    if sigma is None:
        downscale = 2
        sigma = 2 * downscale / 6.0

    LP = zip(
        *[list(_pyramid_laplacian(img, max_layer=K, sigma=sigma)) for img in images]
    )
    F = []

    ilp = 0  # DEBUG
    for lp in LP:
        imageio.imwrite("M_K{}.tif".format(ilp), M.astype(np.uint8) * 120)  # DEBUG

        fused = np.zeros_like(lp[0])
        M = resize(
            M,
            lp[0].shape,
            preserve_range=True,
            order=0,
            mode="edge",
            anti_aliasing=False,
        )
        for (i, l) in zip(range(1, 1 + len(lp)), lp):
            imageio.imwrite(
                "LP_{}_K{}.tif".format(i - 1, ilp), l.astype(np.float32)
            )  # DEBUG
            fused[M == i] = l[M == i]
        F.append(fused)

        ilp += 1  # DEBUG

    ilp = 0  # DEBUG

    fused_F = F[-1]
    for f in reversed(F[:-1]):
        assert all(i <= j for (i, j) in zip(fused_F.shape, f.shape))
        resized_F = resize(fused_F, f.shape, order=1, mode="edge", anti_aliasing=False)
        smoothed_F = _smooth(resized_F, sigma=sigma, mode="reflect", cval=0)
        fused_F = smoothed_F + f

        imageio.imwrite(
            "fused_K{}.tif".format(ilp), fused_F.astype(np.float32)
        )  # DEBUG

        ilp += 1  # DEBUG

    return fused_F


@timeit
@jit
def _generate_seeds(D, t=0.5):
    """
    Find seed pixels by density distributions.

    Parameters
    ----------
    D : np.ndarray
        Density distribution of supplied images.
    t : float
        Threshold for seed pixels, default to 0.5
    """
    S = D[0].copy()
    for d in D[1:]:
        S[d > S] = d[d > S]
    return S > t


def _disk(radius, dtype=np.uint8):
    L = np.arange(-radius + 1, radius)
    X, Y = np.meshgrid(L, L)
    return np.asarray((X ** 2 + Y ** 2) < radius ** 2, dtype=dtype)


@timeit
@jit
def _density_distribution(n, M, r):
    """
    Calculate density distribution along specified circular regions.

    Parameters
    ----------
    n : int
        Number of blurred images.
    M : np.ndarray
        The mask image.
    r : float
        Radius of circle that counted as spatial neighborhood.
    """
    D = []
    selem = _disk(r)
    # normalization term: c
    Ar = np.ones(M.shape, dtype=np.float32)
    c = ndi.convolve(Ar, selem, mode="constant", cval=0)
    for _n in range(1, n + 1):
        # delta function
        Mp = (M == _n).astype(M.dtype)
        v = ndi.convolve(Mp, selem, mode="constant", cval=0)
        Dp = v / c
        D.append(Dp)
    return D


def _modified_laplacian(I):
    """
    Calculate modified Laplacian.

    Parameters
    ----------
    I : np.ndarray
        Supplied raw image.
    """
    ny, nx = I.shape

    J = cp.empty_like(I)
    block_sz = (TILE_WIDTH, TILE_WIDTH)
    grid_sz = (int(ceil(nx / TILE_WIDTH)), int(ceil(ny / TILE_WIDTH)))
    modified_laplacian_kernel(grid_sz, block_sz, (J, I, nx, ny))

    return J


def sml(I, T):
    """
    Calculate summed-modified-Laplacian (SML) measurement.

    Parameters
    ----------
    I : np.ndarray
        Input image.
    T : float
        The discrimination threshold, optimal value of T varies in [0, 10].
    """
    ny, nx = I.shape
    S = _modified_laplacian(I)

    block_sz = (TILE_WIDTH, TILE_WIDTH)
    grid_sz = (int(ceil(nx / TILE_WIDTH)), int(ceil(ny / TILE_WIDTH)))

    sml_kernel(grid_sz, block_sz, (S, S, nx, ny, T))

    return S


@timeit
def _generate_init_mask(I, T):
    """
    Generate mask estimation based on SML.

    Parameters
    ----------
    I : list of np.ndarray
        List of original raw images.
    T : float
        Blur level criteria.
    """
    # temporary buffer for SML
    S = cp.array(I[0])
    nelem = S.size

    n_threads = 1024
    block_sz = (n_threads,)
    grid_sz = (int(ceil(nelem / n_threads)),)

    M, V = cp.ones_like(S, dtype=cp.int32), sml(S, T)
    for i, iI in enumerate(I[1:], 2):
        S = cp.array(iI)
        S = sml(S, T)
        keep_max_kernel(grid_sz, block_sz, (M, V, S, nelem, i))
    return cp.asnumpy(M)


@timeit
def dbrg(images, T, r):
    """
    Segmentation by density-based region growing (DBRG).

    Parameters
    ----------
    n : int
        Number of blurred images.
    M : np.ndarray
        The mask image.
    r : int
        Density connectivity search radius.
    """
    n = len(images)
    M = _generate_init_mask(images, T)
    D = _density_distribution(n, M, r)
    S = _generate_seeds(D)

    # make sure there is at least one seed
    assert S.any()

    # unlabeled
    R = np.full(M.shape, 0, dtype=np.uint32)
    V = np.full(M.shape, np.NINF, dtype=np.float32)

    # label by density map
    for i, d in enumerate(D):
        logger.debug("density {}".format(i))
        R[(d > V) & S] = i + 1
        V[(d > V) & S] = d[(d > V) & S]

    # label by density connectivity
    v = np.empty(len(D) + 1, dtype=np.uint32)  # buffer

    @timeit
    @jit(nopython=True)
    def ps_func(M, R, v):
        n, m = M.shape
        ps = []  # reset of the pixel coordinates
        for y in range(0, n):
            for x in range(0, m):
                if R[y, x] > 0:
                    continue
                pu = min(y + r, n - 1)
                pd = max(y - r, 0)
                pr = min(x + r, m - 1)
                pl = max(x - r, 0)
                v.fill(0)
                for yy in range(pd, pu + 1):
                    for xx in range(pl, pr + 1):
                        if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r:
                            v[R[yy, xx]] += 1
                R[y, x] = v.argmax()
                if R[y, x] == 0:
                    ps.append((y, x))
        return ps

    ps = ps_func(M, R, v)

    # label by nearest neighbor
    @timeit
    @jit(nopython=True)
    def psv_func(ps, M, R):
        n, m = M.shape
        # psv = [] # filled result
        for y, x in ps:
            r = 1
            while True:
                pu = min(y + r, n - 1)
                pd = max(y - r, 0)
                pr = min(x + r, m - 1)
                pl = max(x - r, 0)
                v = []
                for yy in range(pd, pu + 1):
                    for xx in range(pl, pr + 1):
                        if R[yy, xx] > 0:
                            v.append(
                                (R[yy, xx], (xx - x) * (xx - x) + (yy - y) * (yy - y))
                            )
                if len(v) == 0:
                    r += 1
                else:
                    # v.sort(key=lambda p: p[1])
                    # psv.append(v[0][0])
                    R_min, _d_min = v[0]
                    for _R, _d in v[1:]:
                        if _d < _d_min:
                            R_min, _d_min = _R, _d
                    # psv.append(R_min)
                    R[y, x] = R_min
                    break
        # return psv
        return R

    # psv = psv_func(ps, M, R)
    if ps:
        R = psv_func(ps, M, R)

    # move into psv
    # for (y, x), v in zip(ps, psv):
    #    R[y, x] = v

    # make sure each position is assigned a mask value
    assert np.all(R != 0)
    return R


@timeit
def rmlp2(images, T=1 / 255.0, r=4, K=7, sigma=None):
    """
    Perform region-based Laplacian pyramids multi-focus image fusion.

    Parameters
    ----------
    images : list of np.ndarray
        Blurred images.
    T : float
        Initial mask threshold.
    r : int
        Density connectivity search radius.
    K : int
        Level of the pyramids.
    """
    R = dbrg(images, T, r)
    F = pyramid_fusion(images, R, K, sigma=sigma)
    return F
