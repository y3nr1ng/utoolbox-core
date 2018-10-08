import numpy as np

def rm_slices(array, i_rm):
    """Remove slices from the array use the first dimension.

    Parameters
    ----------
    array : ndarray
        Array to remove slices.
    i_rm : tuple of integer
        Denotes the range of slice to remove.
    """
    ind = np.indices(array.shape)[0]
    i_rm = np.hstack([ind[i] for i in range(*i_rm)])
    return np.take(array, sorted(set(ind) - set(i_rm)))
