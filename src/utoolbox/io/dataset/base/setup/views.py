from abc import ABCMeta, abstractmethod

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["MultiViewDataset", "MultiViewDatasetIterator"]

MULTI_VIEW_INDEX_STR = "view"


class MultiViewDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        def load_view_info():
            views = self._load_view_info()
            self._update_inventory_index({MULTI_VIEW_INDEX_STR: views})

        self.register_preload_func(
            load_view_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    ##

    @abstractmethod
    def _load_view_info(self):
        pass


class MultiViewDatasetIterator(DatasetIterator):
    def __init__(self, dataset: MultiViewDataset, **kwargs):
        super().__init__(dataset, index=MULTI_VIEW_INDEX_STR, **kwargs)
