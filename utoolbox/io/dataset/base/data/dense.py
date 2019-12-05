from abc import ABCMeta, abstractmethod

from dask import delayed
import dask.array as da

from ..generic import GenericDataset

__all__ = ["DenseDataset"]


class DenseDataset(GenericDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    ##

    @property
    @abstractmethod
    def read_func(self):
        pass

    ##

    def preload(self):
        shape, dtype = self._load_array_info()

        data_vars = self.dataset.data_vars.keys()
        coords = self.dataset.coords.items()

        for data_var in data_vars:
            # TODO how to generate coords
            file_list = self._retrieve_file_list(data_var, coords)
            data = [
                da.from_delayed(delayed(self.read_func)(file_path), shape, dtype)
                for file_path in file_list
            ]

    ##

    @abstractmethod
    def _load_array_info(self):
        """Load single data var info, in order to rebuild it in Dask."""
        pass

    @abstractmethod
    def _retrieve_file_list(self, data_var, coords):
        pass
