import numpy as np

from .registry import BaseContainer

class Volume(BaseContainer, np.ndarray):
    """Container for volumetric image"""
    def __new__(cls, source=None, **kwargs):
        if source is None:
            return super(Volume, cls).__new__(cls, **kwargs)

        # requires file operation instead of templating
        if not isinstance(source, np.ndarray):
            #TODO use utoolbox.io to determine the proper way to open
            import imageio
            array = imageio.volread(source)

        return array.view(cls)

    def __array_finalize__(self, obj):
        """Fill-in default metadata."""
        # in the middle of __new__
        if obj is None:
            return

        self.resolution = getattr(obj, 'resolution', None)

    @property
    def ndim(self):
        return 3
