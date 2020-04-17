"""
This module provides basic book-keeping for different dataset access types.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

__all__ = ["SessionDataset"]

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

        # TODO provide a contextlib wrapped interface for session objects (re-wrap store -> handle)
        self.register_preload_func(open_session, priority=60)

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    ##

    @abstractmethod
    def _open_session(self):
        """Open session to access a dataset."""
        pass

    @abstractmethod
    def _close_session(self):
        """Close opened session and cleanup."""
        pass
