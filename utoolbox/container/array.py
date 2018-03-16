import logging
logger = logging.getLogger(__name__)

import numpy as np
import imageio

from .registry import BaseContainer

class DenseArray(BaseContainer, np.ndarray):
    """Container for data represented by dense array.

    Reference
    ---------
    imageio.core.util.Image
    """
    def __new__(cls, source=None, **kwargs):
        if source is None:
            obj = super(Volume, cls).__new__(cls, **kwargs)
        else:
            if isinstance(source, np.ndarray):
                obj = source.view(cls)
            else:
                obj = cls._load_externally(source).view(cls)

        return obj

    def __array_finalize__(self, obj):
        if isinstance(obj, Volume):
            # from view-casting
            self._copy_metadata(obj.metadata)
        else:
            # in the middle of __new__ or from templating
            return

    def __array_wrap__ (self, array, context=None):
        """Return a native ndarray when reducting ufunc is applied."""
        if not array.shape:
            logger.debug("__array_wrap__ -> scalar")
            # scalar
            return array.dtype.type(array)
        elif array.shape != self.shape:
            logger.debug("__array_wrap__ -> np.ndarray")
            logger.debug("context={}".format(context))
            # to native ndarray
            return array.view(type=np.ndarray)
        else:
            logger.debug("__array_wrap__ -> utoolbox.container.Volume")
            # remain as utoolbox.container.Volume
            return array

    @staticmethod
    def _load_externally(source):
        raise NotImplementedError

class Image(DenseArray):
    """2-D, single channel image."""
    @staticmethod
    def _load_externally(source):
        #TODO use utoolbox.io to determine the proper way to open
        return imageio.imread(source)

class Volume(DenseArray):
    """3-D, single channel image."""
    @staticmethod
    def _load_externally(source):
        #TODO use utoolbox.io to determine the proper way to open
        return imageio.volread(source)
