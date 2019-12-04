from abc import ABCMeta, abstractmethod

from .generic import GenericDataset

__all__ = ["DenseDataset"]


class DenseDataset(GenericDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    ##

    ##

    @abstractmethod
    def _load_array_info(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass
