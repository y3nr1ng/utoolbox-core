import logging
logger = logging.getLogger(__name__)

import numpy as np

from .registry import BaseContainer

class Volume(BaseContainer, np.ndarray):
    """Container for volumetric image

    Reference
    ---------
    imageio.core.util.Image
    """
    def __new__(cls, source=None, **kwargs):
        if source is None:
            obj = super(Volume, cls).__new__(cls, **kwargs)

        # requires file operation instead of templating
        if isinstance(source, np.ndarray):
            obj = source.view(cls)
        else:
            #TODO use utoolbox.io to determine the proper way to open
            import imageio
            obj = imageio.volread(source).view(cls)

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
