import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from enum import IntEnum
from typing import Callable, Optional, Tuple
from uuid import uuid4

import pandas as pd
from humanfriendly.tables import format_pretty_table

from .error import PreloadError, UnsupportedDatasetError

__all__ = ["BaseDataset", "PreloadPriorityOffset"]

logger = logging.getLogger("utoolbox.io.dataset")


class PreloadPriorityOffset(IntEnum):
    Metadata = 40
    Data = 50
    User = 100


class BaseDataset(metaclass=ABCMeta):
    def __init__(self):
        self._data, self._inventory = dict(), dict()
        self._preload_funcs = []

        def test_readability():
            try:
                self._metadata = self._load_metadata()
                if not self._can_read():
                    raise UnsupportedDatasetError()
            except Exception as e:
                raise UnsupportedDatasetError(str(e))

        def list_files():
            self._files = self._enumerate_files()
            self._files.sort()
            logger.info(f"found {len(self.files)} file(s)")

        self.register_preload_func(test_readability, priority=20)
        self.register_preload_func(list_files, priority=30)
        self.register_preload_func(self._consolidate_inventory, priority=50)

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
    def preload_funcs(self) -> Tuple[Callable[[], None]]:
        """List of preload functions."""
        return tuple(self._preload_funcs)

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

    @classmethod
    def load(cls, *args, **kwargs):
        """
        Load dataset and kickstart preload functions.
        """
        # 1) construct dataset
        ds = cls(*args, **kwargs)

        # 2) populate all the info
        ds.preload()

        return ds

    @classmethod
    def dump(cls, path, dataset=None):
        """
        Dump dataset.

        Args:
            path (str): data destination
            dataset (optional): if provided, serialize the provided dataset instead of
                current object
        """
        raise NotImplementedError("serialization is not supported")

    ##

    def register_preload_func(self, func, priority: Optional[int] = None):
        """
        Register functions to execute during preload steps.
        
        By default,
            - 0-99, internal functions
                - 0-49: metadata
                    - 10, open session to access the dataset
                    - 20, readability test
                    - 30, list files
                    - 40, load dataset metadata
                - 50-99: data
                    - 50, consolidate dataset dimension
                    - 80, assign uuid to data
            - 100-, user functions

        Args:
            func : the function
            priority (optional, int): priority during execution, lower is higher, 
                append to the lowest when not provided
        """
        if priority is None:
            # find any number that is over 50
            pmax = -1
            for p, _ in self._preload_funcs:
                if p >= PreloadPriorityOffset.User and p > pmax:
                    pmax = p
            priority = pmax + 1
            logger.debug(f"auto assign preload priority {priority}")

        self._preload_funcs.append((priority, func))

    def preload(self):
        # sort by priority
        self._preload_funcs.sort(key=lambda f: f[0])
        logger.debug(f"{len(self._preload_funcs)} preload functions registered")

        if logger.getEffectiveLevel() >= logging.DEBUG:
            # dump all the preload functions
            prev_priority = -1
            table = []
            for priority, func in self.preload_funcs:
                if priority != prev_priority:
                    prev_priority = priority
                    prefix_str = int(priority)
                else:
                    prefix_str = " "
                func_name = func.__name__.strip("_")
                table.append([prefix_str, func_name])
            print(format_pretty_table(table, ["Priority", "Function"]))

        logger.info("start preloading")
        for _, func in self.preload_funcs:
            try:
                func()
            except UnsupportedDatasetError:
                # shunt
                raise
            except Exception as e:
                func_name = func.__name__.strip("_")
                raise PreloadError(f'failed at "{func_name}": {str(e)}')

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
