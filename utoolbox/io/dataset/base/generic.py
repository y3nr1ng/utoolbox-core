from abc import ABCMeta, abstractmethod
import logging

import xarray as xr

from .error import UnsupportedDatasetError

__all__ = ["GenericDataset"]

logger = logging.getLogger(__name__)


class GenericDataset(metaclass=ABCMeta):
    def __init__(self):
        self._dataset = xr.Dataset()
        self._metadata = self._load_metadata()

        if not self._can_read():
            raise UnsupportedDatasetError()

    ##

    @property
    def dataset(self):
        return self._dataset

    @property
    def metadata(self):
        return self._metadata

    ##

    @abstractmethod
    def _can_read(self):
        """Whether this dataset can read data from the specified URI."""
        pass

    @abstractmethod
    def _load_metadata(self):
        pass
