import logging
from typing import List

import numpy as np

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

    @property
    def read_func(self):
        def func(uri, shape, dtype):
            # TODO
            pass

        return func

    ##

    @classmethod
    def dump(cls, store: str, dataset):
        pass

    ##

    def _open_session(self):
        pass

    def _close_session(self):
        pass

    def _can_read(self):
        pass

    def _enumerate_files(self):
        pass

    def _load_array_info(self):
        pass

    def _load_channel_info(self):
        pass

    def _load_metadata(self):
        pass

    def _load_timestamp(self) -> List[np.datetime64]:
        pass

    def _load_voxel_size(self):
        pass

    def _retrieve_file_list(self, coord_dict):
        pass


class ImarisStitcherDataset(ImarisDataset, TiledDataset):
    def __init__(self, store: str, level: int = 0):
        pass

    ##

    # TODO override root_dir datastore, root_dir is now collection of session dataset

    ##

    @classmethod
    def dump(cls, store: str, dataset):
        pass

    ##

    def _can_read(self):
        pass

    def _load_metadata(self):
        # TODO load metadata from XML, and later pass on to load from H5 file
        pass

    def _load_mapped_coordinates(self):
        # TODO should we use independent loader for index/coordinates?
        pass
