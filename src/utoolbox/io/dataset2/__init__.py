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
