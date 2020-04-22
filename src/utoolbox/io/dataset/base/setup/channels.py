from abc import ABCMeta, abstractmethod

from ..generic import BaseDataset, PreloadPriorityOffset
from .template import DimensionalDataset

__all__ = ["MultiChannelDataset"]


class MultiChannelDataset(BaseDataset, DimensionalDataset, metaclass=ABCMeta):
    index = ("channel",)

    def __init__(self):
        super().__init__()

        def load_channel_info():
            channels = self._load_channel_info()
            self.inventory.update({self.index[0]: channels})

        self.register_preload_func(
            load_channel_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    ##

    @abstractmethod
    def _load_channel_info(self):
        pass


class SingleChannelDataset(MultiChannelDataset, metaclass=ABCMeta):
    def _load_channel_info(self):
        return ["data"]
