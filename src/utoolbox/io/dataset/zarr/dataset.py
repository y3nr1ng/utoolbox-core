import logging

from ..base import (
    DenseDataset,
    MultiChannelDataset,
    MultiViewDataset,
    TiledDataset,
    SessionDataset,
)

__all__ = ["ZarrDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class ZarrDataset(
    SessionDataset, DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset
):
    @property
    def read_func(self):
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

    def _load_tiling_coordinates(self):
        pass

    def _load_view_info(self):
        pass

    def _load_voxel_size(self):
        pass

    def _retrieve_file_list(self, coord_dict):
        pass
