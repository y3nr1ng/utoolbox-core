import glob
import json
import logging
import os

import numpy as np

from ..base import DenseDataset, TiledDataset

from .error import MissingMetadataError

__all__ = ["MicroManagerV1Dataset"]

logger = logging.getLogger(__name__)


class MicroManagerV1Dataset(DenseDataset, TiledDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir

        super().__init__()

    ##

    @property
    def root_dir(self):
        return self._root_dir

    ##

    def _can_read(self):
        version = self.metadata["MicroManagerVersion"]
        return version.startswith("1.")

    def _load_array_info(self):
        bits = self.metadata["BitDepth"]
        return {8: np.uint8, 16: np.uint16}[bits]

    def _load_metadata(self, metadata_name="metadata.txt"):
        # find all `metadata.txt` and try to open until success
        search_path = os.path.join(self.root_dir, "*", metadata_name)
        for metadata_path in glob.iglob(search_path):
            try:
                with open(metadata_path, "r") as fd:
                    metadata = json.load(fd)
                    return metadata["Summary"]
            except KeyError:
                pass
        else:
            raise MissingMetadataError()

    def _load_tiling_positions(self):
        positions = self.metadata["InitialPositionList"]

        coords = {k: [] for k in ("tile_x", "tile_y", "tile_z")}
        for position in positions:
            coord_dict = position["DeviceCoordinatesUm"]
            try:
                coord_x, coord_y = tuple(coord_dict["XY Stage"])
                coords["tile_x"].append(coord_x)
                coords["tile_y"].append(coord_y)
            except KeyError:
                pass
            try:
                coord_z = coord_dict["Z Stage"][0]
                coords["tile_z"].append(coord_z)
            except KeyError:
                pass
        coords = {k: v for k, v in coords.items()}
        return coords
