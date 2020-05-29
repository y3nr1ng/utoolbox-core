import logging
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["TimeSeriesDataset", "TimeSeriesDatasetIterator"]

logger = logging.getLogger("utoolbox.io.dataset")

TIME_SERIES_INDEX_STR = "time"


class TimeSeriesDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        # use assign_coords to add time coords
        def load_timeseries_info():
            timestamps = self._load_timeseries_info()
            self._update_inventory_index({TIME_SERIES_INDEX_STR: timestamps})

        self.register_preload_func(
            load_timeseries_info, priority=PreloadPriorityOffset.Metadata
        )

    ##

    @property
    def interval(self):
        """Difference in time in seconds between each acquired data array."""
        raise NotImplementedError
        return self._t_interval

    @property
    def idle(self):
        """Amount of idle time in seconds between each acquired data array."""
        raise NotImplementedError
        return self._t_idle

    ##

    @abstractmethod
    def _load_timestamps(self) -> List[np.datetime64]:
        pass

    def _load_timeseries_info(self):
        timestamps = self._load_timestamps()
        try:
            if len(timestamps) <= 1:
                logger.warning("single timepoint is simplified as a fixed dataset")
                return None
        except TypeError:
            # timestamps is None
            return None

        # timestamps should be monotonically increaing
        if any(t1 <= t0 for t1, t0 in zip(timestamps[1:], timestamps[:-1])):
            raise RuntimeError("timestamps are not monotonically increasing")

        # parse interval/idle info
        # TODO

        return timestamps


class TimeSeriesDatasetIterator(DatasetIterator):
    def __init__(self, dataset: TimeSeriesDataset, **kwargs):
        super().__init__(dataset, index=TIME_SERIES_INDEX_STR, **kwargs)
