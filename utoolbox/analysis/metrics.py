import numpy as np

def psnr(reference, sample, max=None):
    """Calculate the PSNR of a sample image.

    Parameters
    ----------
    reference : ndarray
        Reference image.
    sample : ndarray
        Sample image of the same type and shape as the reference image.
    max : integer or float, optional
        Maximum sampling range, default to data type limitation.
    """
    mse = np.mean((sample-reference)**2)
    info = np.finfo if issubclass(reference, np.inexact) else np.iinfo
    eps = info(reference.dtype).eps
    if mse <= eps:
        #TODO fix return value and max boundary
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
