from abc import ABCMeta, abstractmethod
import logging

from ..generic import BaseDataset

__all__ = ["MultiViewDataset"]

logger = logging.getLogger(__name__)


class MultiViewDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        views = self._load_view_info()
        self.inventory.update({"view": views})

    ##

    ##

    @abstractmethod
    def _load_view_info(self):
        pass
