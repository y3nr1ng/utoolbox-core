"""
This module holds all the logical layouts and their corresponding handling
methods.
"""
import abc
import logging
logger = logging.getLogger(__name__)
import warnings

import imageio

def log_warn(message, *args, **kwargs):
    logger.warn(message)

class BaseLayout(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def read(src, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def write(dst, data, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def reduce_to(self):
        pass

    @property
    @abc.abstractmethod
    def expand_to(self):
        pass

class Volume(BaseLayout):
    @staticmethod
    def read(src, **kwargs):
        with warnings.catch_warnings():
            warnings.showwarning = log_warn
            return imageio.volread(src, **kwargs)

    @staticmethod
    def write(dst, data, **kwargs):
        imageio.volwrite(dst, data, **kwargs)

    @property
    def reduce_to(self):
        return Image

    @property
    @abc.abstractmethod
    def expand_to(self):
        raise NotImplementedError

class Image(BaseLayout):
    @staticmethod
    def read(src, **kwargs):
        with warnings.catch_warnings():
            warnings.showwarning = log_warn
            return imageio.imread(src, **kwargs)

    @staticmethod
    def write(dst, data, **kwargs):
        imageio.imwrite(dst, data, **kwargs)

    @property
    def reduce_to(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def expand_to(self):
        raise Volume
