from ..generic import BaseDataset

__all__ = ["TimeSeriesDataset"]


class TimeSeriesDataset(BaseDataset):
    def __init__(self):
        super().__init__()

        # use assign_coords to add time coords
