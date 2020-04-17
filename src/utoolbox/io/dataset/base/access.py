"""
This module provides basic book-keeping for different dataset access types.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

from .generic import BaseDataset

__all__ = ["DirectoryDataset", "SessionDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class DirectoryDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self, root_dir: str):
        super().__init__()

        self._root_dir = root_dir

    ##

    @property
    def root_dir(self) -> str:
        return self._root_dir


class SessionDataset(DirectoryDataset):
    def __init__(self, store: str, path: Optional[str] = None):
        super().__init__(root_dir=store)

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

    ##

    def close(self):
        """
        Explicitly close the dataset session.
        
        If the dataset is not accessed within a with block, please remember to 
        call this method to close the dataset.
        """
        if self._handle is None:
            return
        self._close_session()
        self._handle = None

    ##

    @abstractmethod
    def _open_session(self):
        """Open session to access a dataset."""
        pass

    @abstractmethod
    def _close_session(self):
        """Close opened session and cleanup."""
        pass
