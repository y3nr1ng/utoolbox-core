from abc import ABCMeta, abstractmethod
import logging

from ..generic import BaseDataset

__all__ = ["MultiChannelDataset"]

logger = logging.getLogger(__name__)


class MultiChannelDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        channels = self._load_channel_info()
        self.inventory.update({"channel": channels})

    ##

    ##

    @abstractmethod
    def _load_channel_info(self):
        pass


class SingleChannelDataset(MultiChannelDataset, metaclass=ABCMeta):
    def _load_channel_info(self):
        return ["data"]
