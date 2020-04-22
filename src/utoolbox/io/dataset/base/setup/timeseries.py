from abc import ABCMeta, abstractmethod

from ..generic import BaseDataset, PreloadPriorityOffset
from .template import DimensionalDataset

__all__ = ["TimeSeriesDataset"]


class TimeSeriesDataset(BaseDataset, DimensionalDataset, metaclass=ABCMeta):
    index = ("time",)

    def __init__(self):
        super().__init__()

        # use assign_coords to add time coords
        def load_timeseries_info():
            raise NotImplementedError

        self.register_preload_func(
            load_timeseries_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    @abstractmethod
    def _load_timeseries_info(self):
        pass
