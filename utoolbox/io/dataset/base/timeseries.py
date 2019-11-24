from .generic import GenericDataset

__all__ = ["TimeSeriesDataset"]


class TimeSeriesDataset(GenericDataset):
    def __init__(self):
        super().__init__()

        # use assign_coords to add time coords
