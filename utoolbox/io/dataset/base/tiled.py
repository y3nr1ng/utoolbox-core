from .generic import GenericDataset

__all__ = ["TiledDataset"]


class TiledDataset(GenericDataset):
    def __init__(self):
        super().__init__()
