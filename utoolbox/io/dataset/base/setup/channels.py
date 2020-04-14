from abc import ABCMeta, abstractmethod
import logging

from ..generic import BaseDataset, PreloadPriorityOffset

__all__ = ["MultiChannelDataset"]

logger = logging.getLogger(__name__)


class MultiChannelDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        def load_channel_info():
            channels = self._load_channel_info()
            self.inventory.update({"channel": channels})

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
