# pylint: disable=undefined-variable

from abc import abstractmethod
from collections.abc import Mapping
import os

class Dataset(Mapping):
    def __init__(self, root):
        """
        :param str root: source of the dataset, flat layout
        """
        self._root = root
        self._metadata = self._load_metadata()
        self._datastore = self._load_datastore() 

    def __getitem__(self, key):
        return self._datastore[key]
    
    def __iter__(self):
        return self._datastore
    
    def __len__(self):
        return len(self._datastore)

    @property
    def metadata(self):
        return self._metadata

    @property
    def root(self):
        return self._root

    def to_hdf(self, dst_root=None, virtual=True):
        raise NotImplementedError

    @abstractmethod
    def _load_datastore(self):
        raise NotImplementedError

    def _load_metadata(self):
        pass

class MultiChannelDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
    
    def __iter__(self):
        return iter(self._datastore)

    @abstractmethod
    def _find_channels(self):
        return NotImplementedError

    @abstractmethod
    def _load_channel(self, channel):
        return NotImplementedError

    def _load_datastore(self):
        """Override for multi-channel setup."""
        return {
            channel: self._load_channel(channel)
            for channel in self._find_channels()
        }
        