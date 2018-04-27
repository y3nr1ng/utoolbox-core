import logging
logger = logging.getLogger(__name__)

import imageio

__all__ = [
    'Volume',
]

layouts = {}

class LayoutRegistry(type):
    """Keep a record for all available logical layout of data."""
    def __new__(meta, name, bases, attrs):
        cls = type.__new__(meta, name, bases, attrs)
        layouts[name] = cls
        logger.debug("New layout \"{}\" added.".format(cls))
        return cls

class BaseLayout(metaclass=LayoutRegistry):
    def __str__(self):
        return "Base"

    @staticmethod
    def read(src):
        raise NotImplementedError

    @staticmethod
    def write(dest, data):
        raise NotImplementedError

class Volume(BaseLayout):
    def __str__(self):
        return "Volume"

    @staticmethod
    def read(src):
        return imageio.volread(src)

    @staticmethod
    def write(dest, data):
        imageio.volwrite(dest, data)

class LayeredImage(BaseLayout):
    pass

class Image(BaseLayout):
    def __str__(self):
        return "Image"

    @staticmethod
    def read(src):
        return imageio.imread(src)

    @staticmethod
    def write(dest, data):
        imageio.imwrite(dest, data)

class MultiChannelVolume(BaseLayout):
    pass

class MultiChannelImage(BaseLayout):
    pass
