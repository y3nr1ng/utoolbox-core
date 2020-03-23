from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import logging
from uuid import uuid4

import pandas as pd

from .error import UnsupportedDatasetError

__all__ = ["BaseDataset"]

logger = logging.getLogger(__name__)


class BaseDataset(metaclass=ABCMeta):
    def __init__(self):
        self._data, self._inventory = dict(), dict()

        try:
            self._metadata = self._load_metadata()
            if not self._can_read():
                raise UnsupportedDatasetError()
        except Exception as e:
            raise UnsupportedDatasetError(str(e))

        self._files = self._enumerate_files()
        self._files.sort()
        logger.info(f"found {len(self.files)} file(s)")

    def __getattr__(self, key):
        return self.inventory.__getattr__(key)

    def __getitem__(self, key):
        if isinstance(key, pd.Series):
            if len(key) > 1:
                raise KeyError("multiple keys provided")
            uuid = key.values[0]
        elif isinstance(key, dict):
            # rebuild coordinate
            coord_list = tuple(key[k] for k in self.inventory.index.names)
            uuid = self.inventory.__getitem__(coord_list)
        elif isinstance(key, str):
            uuid = key
        else:
            raise KeyError("unknown key format")
        try:
            return self.data[uuid]
        except KeyError:
            return self._missing_data()

    def __len__(self):
        return self.inventory.__len__()

    ##

    @property
    def data(self):
        return self._data

    @property
    def files(self):
        return self._files

    @property
    def inventory(self):
        return self._inventory

    @property
    def metadata(self):
        return self._metadata

    @property
    def read_func(self):
        """
        Returns:
            callable(URI, SHAPE, DTYPE)

        Note:
            Dataset has to sort the file list itself, since different dataset may have 
            different sorting requirement!
        """
        raise NotImplementedError("dataset is not readable")

    ##

    @staticmethod
    def dump(dataset):
        raise NotImplementedError("serialization method undefined")

    ##

    @abstractmethod
    def _can_read(self):
        """Whether this dataset can read data from the specified URI."""
        pass

    def _consolidate_inventory(self):
        assert self.inventory, "no inventory specification"

        # sort coordinate
        coords = sorted(self.inventory.items(), key=lambda kv: len(kv[1]))
        coords = OrderedDict(coords)

        # generate product index
        index = pd.MultiIndex.from_product(coords.values(), names=coords.keys())
        self._inventory = index

    @abstractmethod
    def _enumerate_files(self):
        pass

    @abstractmethod
    def _load_metadata(self):
        pass

    def _missing_data(self):
        raise KeyError("missing data")

    def _register_data(self, data):
        uuid = str(uuid4())
        self.data[uuid] = data
        return uuid
