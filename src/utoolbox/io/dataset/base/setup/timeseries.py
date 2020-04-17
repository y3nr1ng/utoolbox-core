from ..generic import BaseDataset, PreloadPriorityOffset

__all__ = ["TimeSeriesDataset"]


class TimeSeriesDataset(BaseDataset):
    def __init__(self):
        super().__init__()

        # use assign_coords to add time coords
        def load_timeseries_info():
            pass

        self.register_preload_func(
            load_timeseries_info, priority=PreloadPriorityOffset.Metadata
        )
