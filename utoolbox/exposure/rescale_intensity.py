import logging

import cupy as cp
import numpy as np

__all__ = ["RescaleIntensity"]

logger = logging.getLogger(__name__)


class RescaleIntensity(object):
    """
    Return image after stretching or shrinking its intensity levels. The algorithm is
    based on the implementation in `skimage`_.

    .. _skimage:
        https://scikit-image.org/docs/dev/api/skimage.exposure.html
    """

    def __call__(self, data, in_range="image", out_range="dtype"):
        im, iM = RescaleIntensity.as_min_max(data, in_range)
        om, oM = RescaleIntensity.as_min_max(data, out_range, clip_neg=(im >= 0))

        data = data.clip(im, iM)

        @cp.fuse
        def _ops(data, im, iM, om, oM):
            # normalize
            data = (data - im) / (iM - im)
            # scale to range
            data = data * (oM - om) + om
            return data

        return _ops(data, im, iM, om, oM)

    @staticmethod
    def as_min_max(data, range_value, clip_neg=False):
        """
        Return intensity range based based on desired value type.

        Args:
            data (np.ndarray): Input data.
            range_value (str/np.dtype/(float, float)): The image range is configured by 
                this parameter.
            clip_neg (bool, optional): If True, clip the negative range. 
        
        Returns:
            :rtype: (cp.float32, cp.float32)

        Raises:
            TypeError: If this function cannot resolve the intensity range pair.
        """
        if range_value == "image":
            m, M = np.asscalar(data.min()), np.asscalar(data.max())    
        else:
            if isinstance(range_value, tuple):
                # force extraction tuple
                m, M = range_value
            else:
                if range_value == "dtype":
                    dtype = np.dtype(data.dtype).type
                else:
                    dtype = range_value
                if issubclass(dtype, np.integer):
                    info = np.iinfo(dtype)
                    m, M = info.min, info.max
                elif issubclass(dtype, np.floating):
                    m, M = -1.0, 1.0
                else:
                    raise TypeError("unknown data type")
                m = 0 if clip_neg else m
        m, M = np.float32(m), np.float32(M)
        return m, M


if __name__ == "__main__":
    """
    rescale_intensity = RescaleIntensity()
    arr = cp.array([51, 102, 153], dtype=np.uint8)
    print(arr)
    print(type(arr))
    out = rescale_intensity(arr)
    print(out)
    print(type(out))
    print()

    arr = 1.0 * arr
    print(arr)
    print(type(arr))
    out = rescale_intensity(arr)
    print(out)
    print(type(out))
    print()

    print(arr)
    print(type(arr))
    out = rescale_intensity(arr, in_range=(0, 255))
    print(out)
    print(type(out))
    print()

    print(arr)
    print(type(arr))
    out = rescale_intensity(arr, in_range=(0, 102))
    print(out)
    print(type(out))
    print()

    arr = np.array([51, 102, 153], dtype=np.uint8)
    print(arr)
    print(type(arr))
    out = rescale_intensity(arr, out_range=(0, 127))
    print(out)
    print(type(out))
    print()
    """

    import imageio
    from numpy.testing import assert_array_almost_equal

    image = imageio.volread("cell_in.tif")
    print("{}, {}".format(image.shape, image.dtype))

    from utoolbox.utils.decorator import timeit

    @timeit
    def cpu(image):
        from skimage.exposure import rescale_intensity

        for _ in range(10):
            result = rescale_intensity(image, out_range=(0, 1))
            result, image = image, result
        return image

    result1 = cpu(image)

    @timeit
    def gpu(image):
        rescale_intensity = RescaleIntensity()
        image = cp.asarray(image)
        for _ in range(10):
            result = rescale_intensity(image, out_range=(0, 1))
            result, image = image, result
        return cp.asnumpy(image)

    result2 = gpu(image)
    imageio.volwrite("cell_out.tif", result2)

    assert_array_almost_equal(result1, result2)
