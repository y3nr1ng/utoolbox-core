import logging
from typing import Optional

import zarr

from ..base import (
    DenseDataset,
    MultiChannelDataset,
    MultiViewDataset,
    SessionDataset,
    TiledDataset,
    TimeSeriesDataset,
)

__all__ = ["ZarrDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class ZarrDataset(
    SessionDataset, DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset
):
    version = 1

    """
    Using Zarr directory as backend. 

    Default internal path is '/', post-processing steps use groups with '_' 
    prefix as their root.

    Args:
        store (str): path to the data store
        path (str, optional): group path
    """

    def __init__(self, store: str, path: str = "/"):
        super().__init__(store, path)

    ##

    @property
    def read_func(self):
        pass

    ##

    @classmethod
    def dump(
        cls, store: str, dataset, path: Optional[str] = None, overwrite=False, **kwargs
    ):
        """
        Dump dataset.

        Args:
            store (str): path to the data store
            dataset : serialize the provided dataset
            path (str, optional): internal path
            overwrite (bool, optional): overwrite the dataset if exists
            **kwargs : additional argument for `zarr.open` function
        """
        kwargs["mode"] = "a"
        root = zarr.open(store, **kwargs)

        # test attributes
        root.attrs["zarr_dataset"] = "ZarrDataset"
        root.attrs["format_version"] = cls.version

        if path:
            # nested group
            mode = "w" if overwrite else "w-"
            root = root.open_group(path, mode=mode)

        # TODO start populating the container structure
        if issubclass(dataset, TimeSeriesDataset):
            pass
        if issubclass(dataset, MultiChannelDataset):
            pass
        if issubclass(dataset, (MultiViewDataset, TiledDataset)):
            pass

    ##

    def _open_session(self):
        z = zarr.open(self.root_dir, mode="r")  # don't create it
        self._handle = z[self.path]

        # preview the internal structure
        if logger.getEffectiveLevel() <= logging.DEBUG:
            zarr.tree(self._handle)

    def _close_session(self):
        self._handle.close()
        self._handle = None

    def _can_read(self):
        try:
            magic = self.handle["zarr_dataset"]
            version = self.handle["format_version"]
        except KeyError:
            return False
        else:
            logger.debug(f"a uToolbox written dataset, {magic} {version}")
            return True

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
