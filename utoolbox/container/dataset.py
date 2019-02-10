from abc import ABCMeta, abstractmethod
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

    @abstractmethod
    def _load_datastore(self):
        raise NotImplementedError

class DatasetError(Exception):
    """Base class for dataset-related exceptions."""

class UndefinedConversionError(DatasetError):
    """Raised when dataset conversion is impossible."""