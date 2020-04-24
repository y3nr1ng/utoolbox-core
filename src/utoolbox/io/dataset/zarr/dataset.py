import logging
from collections import OrderedDict
from typing import Optional
from itertools import product

import zarr

from ..base import (
    DenseDataset,
    MultiChannelDataset,
    MultiChannelDatasetIterator,
    MultiViewDataset,
    MultiViewDatasetIterator,
    SessionDataset,
    TiledDataset,
    TiledDatasetIterator,
    TimeSeriesDataset,
    TimeSeriesDatasetIterator,
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

        # start populating the container structure
        #   /time/channel/setup/level
        # welp, i have no idea how to do this cleanly without nested structure
        for i_t, (t, t_dataset) in enumerate(TimeSeriesDatasetIterator(dataset)):
            t_root = root[f"t{i_t}"]
            t_root.attrs["timestamp"] = t

            for i_c, (c, c_dataset) in enumerate(
                MultiChannelDatasetIterator(t_dataset)
            ):
                c_root = t_root[f"c{i_c}"]
                c_root.attrs["channel"] = c

                i_s = 0
                for sv, sv_dataset in MultiViewDatasetIterator(c_dataset):
                    s_root = c_root[f"s{i_s}"]
                    if sv is not None:
                        # write multi-view info
                        s_root.attrs["view"] = sv
                    for st, st_dataset in TiledDatasetIterator(sv_dataset):
                        st_root = sv_root["s"]
                        # TODO fuck

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
