# pylint: disable=undefined-variable

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
import os

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, root):
        """
        Parameters
        ----------
        root : str
            Source of the dataset, flat layout.
        """
        self._root = root
        self._datastore = None
    
    @property
    def datastore(self):
        return self._datastore

    @property
    def root(self):
        return self._root

    @staticmethod
    def from_dataset(ds, **kwargs):
        """
        Convert other form of dataset into current one.
        
        Parameter
        ---------
        ds : AbstractDataset
            Source dataset.
        """
        raise UndefinedConversionError

    @abstractmethod
    def _generate_inventory(self):
        raise NotImplementedError

class AbstractMultiChannelDataset(AbstractDataset, Mapping):
    def __init__(self, root):
        super(AbstractMultiChannelDataset, self).__init__(root)
        self._datastore = dict()

    def __getitem__(self, key):
        return self._datastore[key]
    
    def __iter__(self):
        """During iterations, we are actually iterate over the datastore."""
        return self._datastore
    
    def __len__(self):
        return len(self._datastore)

    # alias
    channels = datastore

    @abstractmethod
    def _map_channels(self):
        """Map channels to datastore from dataset root."""
        return NotImplementedError

class DatasetError(Exception):
    """Base class for dataset exceptions."""

class UndefinedConversionError(DatasetError):
    """Raised when dataset conversion is impossible."""