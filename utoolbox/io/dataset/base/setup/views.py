from abc import ABCMeta, abstractmethod
import logging

from ..generic import BaseDataset, PreloadPriorityOffset

__all__ = ["MultiViewDataset"]

logger = logging.getLogger(__name__)


class MultiViewDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        def load_view_info():
            views = self._load_view_info()
            self.inventory.update({"view": views})

        self.register_preload_func(
            load_view_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    ##

    @abstractmethod
    def _load_view_info(self):
        pass
