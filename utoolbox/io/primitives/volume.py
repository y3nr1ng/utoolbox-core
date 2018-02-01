import logging
logger = logging.getLogger(__name__)

import os.path

import numpy as np
import imageio

class SimpleVolume(np.ndarray):
    def __new__(cls, path=None, array=None, shape=None, dtype=None):
        """SimpleVolume(path=None, array=None, shape=None, dtype=None)

        A subclass of np.ndarray that acts as the common ground for standard
        volumetric dataset.

        Parameters
        ----------
        """
        if path is None and array is None:
            # create the ndarray instance of our type
            obj = super(SimpleVolume, cls).__new__(cls,
                                                   shape=shape, dtype=dtype)
            return obj
        elif path:
            if array is not None:
                logger.warning('\'array\' is shadowed by \'path\', ignored.')
            array = imageio.volread(path)
        # simple view casting
        obj = array.view(cls)
        return obj

    def __array_finalize__(self, obj):
        # fill in default metadata here
        pass

class SIVolume(object):
    """SIVolume()

    A subclass of np.ndarray that acts as the common ground for structured
    illuminated volumetric dataset, where a single SI volume is composed of
    multiple simple volumes.

    Parameters
    ----------
    """
    pass
