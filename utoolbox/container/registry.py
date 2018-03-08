import logging
logger = logging.getLogger(__name__)

containers = {}

class ContainerRegistry(type):
    """Keep a record for all available data containers."""
    def __new__(meta, name, bases, attrs):
        cls = type.__new__(meta, name, bases, attrs)
        containers[name] = cls
        logger.debug("New container \"{}\" added.".format(cls))
        return cls

class BaseContainer(metaclass=ContainerRegistry):
    def __init__(self, *args, resolution=None, **kwargs):
        """
        Parameters
        ----------
        resolution : tuple or list
            Resolution of the unit elements in the container.
        """
        self.resolution = resolution

    @property
    def ndim(self):
        raise NotImplementedError

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        if resolution is None:
            resolution = tuple([1.] * self.ndim)
        elif isinstance(resolution, list):
            resolution = tuple(resolution)
        elif not isinstance(resolution, tuple):
            raise ValueError("invalid resolution parameter")
        self._resolution = resolution
