from abc import ABCMeta, abstractmethod

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["TimeSeriesDataset", "TimeSeriesDatasetIterator"]

TIME_SERIES_INDEX = "time"


class TimeSeriesDataset(BaseDataset, metaclass=ABCMeta):
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


class TimeSeriesDatasetIterator(DatasetIterator):
    def __init__(self, dataset: TimeSeriesDataset, **kwargs):
        super().__init__(dataset, index=TIME_SERIES_INDEX, **kwargs)
