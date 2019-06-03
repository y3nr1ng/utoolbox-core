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
        on_device, dtype = isinstance(data, cp.ndarray), data.dtype
        data = cp.asarray(data)

        im, iM = RescaleIntensity.as_min_max(data, in_range)
        om, oM = RescaleIntensity.as_min_max(data, out_range, clip_neg=(im >= 0))

        data = data.clip(im, iM)

        @cp.fuse
        def _normalize(data, m, M):
            return (data - m) / (M - m)

        data = _normalize(data, im, iM)

        @cp.fuse
        def _scale_to_range(data, m, M):
            return data * (M - m) + m

        data = _scale_to_range(data, om, oM)

        data = data if on_device else cp.asnumpy(data)
        return data.astype(dtype)

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
            :rtype: (float, float)

        Raises:
            TypeError: If this function cannot resolve the intensity range pair.
        """
        if range_value == "image":
            m, M = data.min(), data.max()
            m, M = m.astype(cp.float32), M.astype(cp.float32)
        elif range_value == "dtype":
            dtype = np.dtype(data.dtype).type
            if issubclass(dtype, np.integer):
                info = np.iinfo(dtype)
                m, M = info.min, info.max
            elif issubclass(dtype, np.floating):
                m, M = -1.0, 1.0
            else:
                raise TypeError("unknown data type")
            m = 0 if clip_neg else m
            m, M = cp.float32(m), cp.float32(M)
        else:
            # force extraction tuple
            m, M = range_value
            m, M = cp.float32(m), cp.float32(M)
        # guarantee they are floats
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
    image = imageio.volread('cell_in.tif')
    print(image.shape)

    from utoolbox.util.decorator import timeit
    
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

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    print(mempool.used_bytes())              
    print(mempool.total_bytes())             
    print(pinned_mempool.n_free_blocks())    

