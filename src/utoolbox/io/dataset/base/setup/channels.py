from abc import ABCMeta, abstractmethod

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["MultiChannelDataset", "MultiChannelDatasetIterator"]

MULTI_CHANNEL_INDEX_STR = "channel"


class MultiChannelDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        def load_channel_info():
            channels = self._load_channel_info()
            self._update_inventory_index({MULTI_CHANNEL_INDEX_STR: channels})

        self.register_preload_func(
            load_channel_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    ##

    @abstractmethod
    def _load_channel_info(self):
        pass


class MultiChannelDatasetIterator(DatasetIterator):
    def __init__(self, dataset: MultiChannelDataset, **kwargs):
        super().__init__(dataset, index=MULTI_CHANNEL_INDEX_STR, **kwargs)
