from abc import ABCMeta, abstractmethod

from ..generic import BaseDataset, PreloadPriorityOffset
from .template import DimensionalDataset

__all__ = ["MultiViewDataset"]


class MultiViewDataset(BaseDataset, DimensionalDataset, metaclass=ABCMeta):
    index = ("view",)

    def __init__(self):
        super().__init__()

        def load_view_info():
            views = self._load_view_info()
            self.inventory.update({self.index[0]: views})

        self.register_preload_func(
            load_view_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    ##

    @abstractmethod
    def _load_view_info(self):
        pass
