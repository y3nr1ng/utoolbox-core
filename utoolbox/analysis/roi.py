from types import GeneratorType

from skimage.segmentation import find_boundaries
from skimage.morphology import remove_small_objects

from scipy.ndimage.morphology import binary_fill_holes

from utoolbox.segmentation import chan_vese

def _extract_mask(data, mu=1e-2, tol=1e-4, max_iter=500, dt=1., min_size=256,
                  **kwargs):
    phi = kwargs.pop('phi', 'checkerboard')
    mask, phi, _  = chan_vese(
        data,
        mu=mu, tol=tol, max_iter=max_iter, dt=dt, init_level_set=phi,
        extended_output=True
    )

    # fix morphology
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size, in_place=True)

    return mask, phi

def extract_mask(data, **kwargs):
    """
    Extract cellular contour of provided data.

    It will try to yield over the data first.

    Parameters
    ----------
    iterative : bool
        Initialize level set from previous result or not.
    min_size : int
        Minimum size of the feature, default is 256.
    """
    iterative = kwargs.pop('iterative', True)
    if isinstance(data, GeneratorType):
        mask, phi = _extract_mask(next(data), **kwargs)
        masks = [mask]
        for d in data:
            if not iterative:
                phi = 'checkerboard'
            mask, phi = _extract_mask(d, phi=phi, **kwargs)
            masks.append(mask)
        return masks
    else:
        mask, _ = _extract_mask(data, **kwargs)
        return mask

def mask_to_contour(data):
    """Thin wrapper to extract contour using skimage."""
    if isinstance(data, list):
        return [find_boundaries(m, mode='outer') for m in data]
    else:
        return find_boundaries(data, mode='outer')
