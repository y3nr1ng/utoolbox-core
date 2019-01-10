import logging

import numpy as np
from scipy import ndimage as ndi

__all__ = [
    'rmlp'
]

logger = logging.getLogger(__name__)

def _smooth(data, sigma, mode, cval):
    pass

def _pyramid_laplacian(I, max_layer=-1, downscale=2, sigma=None, mode='reflect', cval=0):
    """
    Create laplacian pyramid by yielding them one-by-one.

    Parameters
    ----------
    I : np.ndarray
        Original data.
    max_layer : int, optional
        Number of layers for the pyramid, 0th layer is the original. Default is -1, which builts all possible layers.
    downscale : float, optional
        Downscale factor, default is 2 (in-half).
    sigma : float, optional
        Sigma for the smooth filter. 
    """
    assert(downscale > 1)
    
    if sigma is None:
        # determine sigma that covers 99% of distribution
        sigma = 2*downscale/6. 
    
    layer = 0
    curr_shape = 

def pyramid_fusion(ds, M, K, sigma=None):
    """
    Fuse pyramid layers using the mask.

    Parameters
    ----------
    ds : list of np.ndarray
        The blurred source images.
    M : np.ndarray
        The masked image.
    K : int
        Level of the pyramid.
    sigma : float
        Sigma of the smooth filter.
    """
    if sigma is None:
        downscale = 2
        sigma = 2*downscale / 6.
    
    LP = zip(*[])
    F = []

def _generate_seeds(D, t=.5):
    """
    Find seed pixels by density distributions.

    Parameters
    ----------
    D : np.ndarray
        Density distribution of supplied images.
    t : float   
        Threshold for seed pixels, default to .5
    """
    return np.maximum.reduce(D) > t

def _sphere(r, dtype=np.uint8):
    L = np.arange(-r+1, r)
    X, Y, Z = np.meshgrid(L, L, L)
    return np.asarray((X**2 + Y**2 + Z**2) < r**2, dtype=dtype)

def _density_distribution(n, M, r):
    """
    Calculate density distribution along a circular regions.

    Parameters
    ----------
    n : int
        Number of blurred images.
    M : np.ndarray  
        The mask array.
    r : float
        Radius of circle that counted as spatial neighborhood.
    """
    D = []
    selem = _sphere(r)
    
    Ar = np.ones_like(M)
    c = ndi.convolve(Ar, selem, mode='constant', cval=0) # normalization
    for _n in range(1, n+1):
        logger.debug("density @ level={}".format(_n))
        # delta function
        Mp = (M == _n).astype(int)
        v = ndi.convolve(Mp, selem, mode='constant', cval=0)
        Dp = v / c 
        D.append(Dp)
    return D

def _modified_laplacian(I):
    J = np.empty_like(I)
    n, m, l = I.shape
    for z in range(0, n):
        logger.debug("modified L @ z={}".format(z))
        for y in range(0, m):
            for x in range(0, l):
                pc = I[z, y, x]
                # symmetric padding
                pu = I[z, y, x+1] if x < l-1 else I[z, y, x-1]
                pd = I[z, y, x-1] if x > 0 else I[z, y, x+1]
                pr = I[z, y+1, x] if y < m-1 else I[z, y-1, x]
                pl = I[z, y-1, x] if y > 0 else I[z, y+1, x]
                pt = I[z+1, y, x] if z < n-1 else I[z-1, y, x]
                pb = I[z-1, y, x] if z > 0 else I[z+1, y, x]

                J[z, y, x] = abs(2*pc-pu-pd) + abs(2*pc-pl-pr) + abs(2*pc-pt-pb)
    return J

def sml(I, T):
    """
    Summed-modified-Laplacian (SML) measurement.

    Parameters
    ----------
    I : np.ndarray
        Image to estimate.
    T : float
        The discrimination threshold, optimal value ranges in [0, 10].
    """
    G = _modified_laplacian(I)
    Gp = np.zeros((3, 3, 3)) # buffer

    S = np.empty_like(I)
    n, m, l = S.shape
    for z in range(0, n):
        for y in range(0, m):
            for x in range(0, l):
                pu = min(x+1, l-1)
                pd = max(x-1, 0)
                pr = min(y+1, m-1)
                pl = max(y-1, 0)
                pt = min(z+1, n-1)
                pb = max(z-1, 0)
                Gp = G[pb:pt+1, pl:pr+1, pd:pu+1]

                S[z, y, x] = np.sum(Gp[Gp >= T])
    return S

def _generate_init_mask(ds, T):
    """
    Generate mask estimation based on SML.
    
    Parameters
    ----------
    ds : list of np.ndarray
        Blurred dataset.
    T : float
        Blur level criteria.
    """
    S = [sml(I, T) for I in ds] # TODO: merge in the loop

    M = np.zeros_like(ds[0], dtype=np.uint32)
    V = np.full_like(M, np.NINF)
    for i, s in enumerate(S):
        M[abs(s) > V] = i+1
        V[abs(s) > V] = s[abs(s) > V]
    return M

def dbrg(ds, T, r):
    """
    Segmentation by density-based region growing (DBRG).

    Parameters
    ----------
    ds : np.ndarray
        The mask image.
    T : float
        Initial mask threshold.
    r : int
        Density connectivity search radius.
    """
    M = _generate_init_mask(ds, T)
    D = _density_distribution(len(ds), M, r)
    S = _generate_seeds(D)

    # make sure at least one seed exists
    assert(S.any())

    # unlabeled
    R = np.zeros_like(M, dtype=np.uint32)
    V = np.full_like(M, np.NINF, dtype=np.float32)

    # label by density map
    for i, d in enumerate(D):
        R[(d > V) & S] = i+1
        V[(d > V) & S] = d[(d > V) & S]

    # label by density connectivity
    n, m, l = M.shape
    v = np.empty(len(D)+1, dtype=np.float32)
    ps = [] # reset pixel coordinates
    for z in range(0, n):
        for y in range(0, m):
            for x in range(0, l):
                if R[z, y, x] > 0:
                    continue
                pu = min(x+1, l-1)
                pd = max(x-1, 0)
                pr = min(y+1, m-1)
                pl = max(y-1, 0)
                pt = min(z+1, n-1)
                pb = max(z-1, 0)
                
                v.fill(0)
                for zz in range(pb, pt+1):
                    for yy in range(pl, pr+1):
                        for xx in range(pd, pu+1):
                            if ((xx-x)**2 + (yy-y)**2 + (zz-z)*2 <= r*r):
                                v[R[zz, yy, xx]] += 1
                R[z, y, x] = v.argmax()
                if R[z, y, x] == 0:
                    ps.append((z, y, x))
    
    # label by nearest neighbor
    psv = [] # filled result
    for z, y, x in ps:
        r = 1
        while True:
            pu = min(x+1, l-1)
            pd = max(x-1, 0)
            pr = min(y+1, m-1)
            pl = max(y-1, 0)
            pt = min(z+1, n-1)
            pb = max(z-1, 0)
            v = []
            for zz in range(pb, pt+1):
                    for yy in range(pl, pr+1):
                        for xx in range(pd, pu+1):
                            v.append((R[zz, yy, xx], (xx-x)**2 + (yy-y)**2 + (zz-z)**2))
            if len(v) == 0:
                r += 1
            else:
                v.sort(key=lambda p: p[1])
                psv.append(v[0][0])
                break
    for (z, y, x), v in zip(ps, psv):
        R[z, y, x] = v
    
    # make sure each position is assigned a mask value
    assert(np.all(R != 0))
    return R

def rmlp(ds, T=1/255., r=4, K=None):
    """
    Region-based Laplacian pyramids multi-focus fusion.

    Parameters
    ----------
    ds : list of np.ndarray
        Blurred dataset.
    T : float
        Initial mask threshold.
    r : int
        Density connectivity search radius.
    K : int
        Level of the pyramids, default to `floor(log_2(dim(ds)))`
    """
    # determine K based on data dimension
    if not K:
        pass
        logger.debug("estimate K={}".format(K))
    R = dbrg(ds, T, r)
    F = pyramid_fusion(ds, R, K)
    return F