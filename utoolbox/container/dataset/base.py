# pylint: disable=undefined-variable

from abc import ABC, abstractmethod
from collections.abc import Mapping
import os

from .error import UndefinedConversionError

class AbstractDataset(ABC):
    def __init__(self, root):
        """
        :param str root: source of the dataset, flat layout
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
        
        :param ds: source dataset
        :type ds: :class:`.AbstractDataset`
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
        """
        .. note::
            During iterations, we are actually iterate over the datastore.
        """
        return self._datastore
    
    def __len__(self):
        return len(self._datastore)

    @property
    def channels(self):
        return self._datastore
    
    @property
    def datastore(self):
        """Block the usage to avoid confusion."""
        raise NotImplementedError("use `channels` to retrieve datastores")

    @abstractmethod
    def _map_channels(self):
        """Map channels to datastore from dataset root."""
        return NotImplementedError
