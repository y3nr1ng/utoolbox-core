

class ThresholdSauvola(object):
    """
    Applies Sauvola local threshold to an array.

    Args:
        windows_size (int, optional): Window size specified as a single odd integer.
        k (float, optional): Value of the positive parameter.
        r (float, optional): Dynamic range of standard deviation. If None, set to half 
            of the image dtype range.
    """
    def __call__(self, x, window_size=15, k=0.2, r=None):
        if r is None:
            info = np.iinfo(x.dtype)
            m, M = info.min, info.max
            r = .5 * (M-m)
        a, s = x.mean(), x.std()
        return a * (1 + k * ((s/r)-1.)) 
        