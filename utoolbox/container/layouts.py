"""
This module holds all the logical layouts and their corresponding handling
methods.
"""
import abc
import logging
logger = logging.getLogger(__name__)

import imageio

class BaseLayout(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def read(src):
        pass

    @staticmethod
    @abc.abstractmethod
    def write(dst, data):
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
    def read(src):
        return imageio.volread(src)

    @staticmethod
    def write(dst, data):
        imageio.volwrite(dst, data)

    @property
    def reduce_to(self):
        return Image

    @property
    @abc.abstractmethod
    def expand_to(self):
        raise NotImplementedError

class Image(BaseLayout):
    @staticmethod
    def read(src):
        return imageio.imread(src)

    @staticmethod
    def write(dst, data):
        imageio.imwrite(dst, data)

    @property
    def reduce_to(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def expand_to(self):
        raise Volume