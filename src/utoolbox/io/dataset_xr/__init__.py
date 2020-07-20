"""
Structural design is strongely influenced by `imageio`.

`imageio` is a Python library that provides an easy interface to read and write a wide 
range of image data.
"""
# flake8: noqa

from .dataset import DatasetFormatManager

# instantiate format manager as singleton
formats = DatasetFormatManager()

# load the functions
# TODO

# load all the concrete dataset reader/writer implementations
from . import datasets

# namespace cleanup
del DatasetFormatManager
