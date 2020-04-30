from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["TimeSeriesDataset", "TimeSeriesDatasetIterator"]

TIME_SERIES_INDEX = "time"


class TimeSeriesDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        # use assign_coords to add time coords
        def load_timeseries_info():
            pass  # raise NotImplementedError

        self.register_preload_func(
            load_timeseries_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    @property
    def interval(self):
        """Difference in time in seconds between each acquired data array."""
        return self._t_interval

    @property
    def idle(self):
        """Amount of idle time in seconds between each acquired data array."""
        return self._t_idle

    ##

    @abstractmethod
    def _load_timestamps(self) -> List[np.datetime64]:
        pass

    def _load_timeseries_info(self):
        timestamps = self._load_timestamps()
        # ensure the timestamp has millisecond resolution
        timestamps = [np.datetime64(timestamp, "ms") for timestamp in timestamps]

        dt = timestamps[1:] - timestamps[:-1]
        print(dt)
        raise RuntimeError("DEBUG")


class TimeSeriesDatasetIterator(DatasetIterator):
    def __init__(self, dataset: TimeSeriesDataset, **kwargs):
        super().__init__(dataset, index=TIME_SERIES_INDEX, **kwargs)
