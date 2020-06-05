import logging

from ..base import (
    DenseDataset,
    MultiChannelDataset,
    SessionDataset,
    TiledDataset,
    TimeSeriesDataset,
)

__all__ = ["ImarisDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class ImarisDataset(
    SessionDataset, DenseDataset, MultiChannelDataset, TimeSeriesDataset
):
    """
    Imaris 5 open file format.

    Args:
        store (str): path to the data store
        level (int, optional): resolution level
    """

    def __init__(self, store: str, level: int = 0):
        pass


class ImarisStitcherDataset(ImarisDataset, TiledDataset):
    def __init__(self, store: str, level: int = 0):
        pass

    ##

    # TODO override root_dir datastore, root_dir is now collection of session dataset
