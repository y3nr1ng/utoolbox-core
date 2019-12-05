import glob
import json
import logging
import os

import imageio
import numpy as np

from ..base import DenseDataset, MultiChannelDataset, TiledDataset

from .error import MissingMetadataError

__all__ = ["MicroManagerV1Dataset"]

logger = logging.getLogger(__name__)


class MicroManagerV1Dataset(DenseDataset, MultiChannelDataset, TiledDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir

        super().__init__()

        self.preload()

    ##

    @property
    def read_func(self):
        return imageio.imread

    @property
    def root_dir(self):
        return self._root_dir

    ##

    def _can_read(self):
        version = self.metadata["MicroManagerVersion"]
        return version.startswith("1.")

    def _enumerate_files(self):
        search_path = os.path.join(self.root_dir, "*", "*.tif")
        return glob.glob(search_path)

    def _load_array_info(self):
        # shape
        shape = self.metadata["Height"], self.metadata["Width"]
        nz = self.metadata["Slices"]
        if nz > 1:
            shape = (nz,) + shape

        # dtype
        bits = self.metadata["BitDepth"]
        dtype = {8: np.uint8, 16: np.uint16}[bits]

        return shape, dtype

    def _load_channel_info(self):
        return self.metadata["ChNames"]

    def _retrieve_file_list(self, data_var, coords):
        pass

    def _load_metadata(self, metadata_name="metadata.txt"):
        # find all `metadata.txt` and try to open until success
        search_path = os.path.join(self.root_dir, "*", metadata_name)
        for metadata_path in glob.iglob(search_path):
            try:
                with open(metadata_path, "r") as fd:
                    metadata = json.load(fd)
                    logger.info(f'found metadata at "{metadata_path}"')
                    return metadata["Summary"]
            except KeyError:
                pass
        else:
            raise MissingMetadataError()

    def _load_tiling_positions(self):
        positions = self.metadata["InitialPositionList"]

        coords = {k: [] for k in ("tile_x", "tile_y")}
        for position in positions:
            coord_x, coord_y = tuple(position["DeviceCoordinatesUm"]["XY Stage"])
            coords["tile_x"].append(coord_x)
            coords["tile_y"].append(coord_y)

        # NOTE use `set` to ensure no duplicate items
        coords = {
            k: np.array(list(set(v)), dtype=np.float32) for k, v in coords.items()
        }
        return coords
