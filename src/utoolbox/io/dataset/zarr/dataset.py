import logging

from ..base import DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset

__all__ = ["ZarrDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class ZarrDataset(DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset):
    def __init__(self, root_dir):
        super().__init__()

        self._root_dir = root_dir

        # init internal attribute
        self._handle = None

        def open_zarr_store():
            self._handle = None

        self.register_preload_func(open_zarr_store, priority=60)

    ##

    @property
    def handle(self):
        if self._handle is None:
            raise RuntimeError("dataset is not properly opened")
        return self._handle

    @property
    def read_func(self):
        pass

    @property
    def root_dir(self):
        return self._root_dir

    ##

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
