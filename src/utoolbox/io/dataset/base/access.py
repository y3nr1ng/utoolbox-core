"""
This module provides basic book-keeping for different dataset access types.
"""
import logging
import os
from abc import ABCMeta, abstractmethod

from .generic import BaseDataset

__all__ = ["DirectoryDataset", "SessionDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class DirectoryDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self, root_dir: str):
        super().__init__()

        # NOTE most scripts only uses ~, expandvars is not needed
        root_dir = os.path.expanduser(root_dir)
        if not os.path.exists(root_dir):
            raise ValueError(f'"{root_dir}" does not exist')
        self._root_dir = root_dir

    ##

    @property
    def root_dir(self) -> str:
        return self._root_dir

    ##

    @classmethod
    def dump(cls, root_dir: str, dataset):
        """
        Dump dataset.

        Args:
            root_dir (str): data destination
            dataset : serialize the provided dataset
        """
        raise NotImplementedError("serialization is not supported")


class SessionDataset(DirectoryDataset):
    """
    Dataset that requires a session to access its contents. Since session has a 
    life-cycle, one should use a `with` block to ensure its closure or 
    explicitly call the `close` method after finishing it.

    Args:   
        store (str): path to the data store
        path (str): internal path
    """

    def __init__(self, store: str, path: str):
        super().__init__(root_dir=store)
        self._path = path

        self._handle = None

        def open_session():
            self._open_session()

        self.register_preload_func(open_session, priority=10)

    def __enter__(self):
        self._open_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    ##

    @property
    def handle(self):
        assert (
            self._handle is not None
        ), "dataset session is not opened, programmer error"
        return self._handle

    @property
    def path(self) -> str:
        """Internal path of the session object."""
        return self._path

    ##

    @classmethod
    def dump(cls, store: str, path: str, dataset):
        """
        Dump dataset.

        Args:
            store (str): path to the data store
            path (str): internal path
            dataset : serialize the provided dataset
        """
        raise NotImplementedError("serialization is not supported")

    ##

    def close(self):
        """
        Explicitly close the dataset session.
        
        If the dataset is not accessed within a with block, please remember to 
        call this method to close the dataset.
        """
        if self._handle is None:
            logger.warning(f"dataset is prematurely closed")
        self._close_session()
        self._handle = None

    ##

    @abstractmethod
    def _open_session(self):
        """Open session to access a dataset."""
        pass

    @abstractmethod
    def _close_session(self):
        """
        Close opened session and cleanup.
        
        Note:
            Implementation should set handle to None to signal the closure is 
            completed.
        """
        pass
