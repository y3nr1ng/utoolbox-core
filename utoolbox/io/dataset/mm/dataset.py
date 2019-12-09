import glob
import json
import logging
from operator import itemgetter
import os

from dask import delayed
import dask.array as da
import imageio
import numpy as np
from sparse import COO

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
        def func(uri, shape, dtype):
            # layered volume
            nz, shape = shape[0], shape[1:]
            array = da.stack(
                [
                    da.from_delayed(
                        delayed(imageio.imread, pure=True)(file_path), shape, dtype
                    )
                    for file_path in uri
                ]
            )
            if array.shape[0] != nz:
                logger.warning(f"retrieved layer mis-matched")
            return array

        return func

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

    def _missing_data(self):
        shape, dtype = self._load_array_info()
        return delayed(np.zeros)(shape, dtype)

    def _retrieve_file_list(self, coord_dict):
        prefix = self._tile_prefix[itemgetter("tile_x", "tile_y")(coord_dict)]
        return glob.glob(
            os.path.join(self.root_dir, prefix, f"*_{coord_dict['channel']}_*.tif")
        )

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

    def _load_tiling_coordinates(self):
        positions = self.metadata["InitialPositionList"]

        coords = {k: [] for k in ("tile_x", "tile_y")}
        labels = dict()
        for position in positions:
            # coordinate
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

            # label
            # NOTE MicroManager only tiles in 2D, no need to include Z for indexing
            labels[(np.float32(coord_x), np.float32(coord_y))] = position["Label"]

        # internal bookkeeping
        self._tile_prefix = labels

        return {k: np.array(v, dtype=np.float32) for k, v in coords.items()}

    def _load_tiling_info(self):
        index, coords = super()._load_tiling_info()
        try:
            # NOTE MicroManger does not have Z tiling, drop it if it exists
            del index["tile_z"]
        except KeyError:
            pass

        return index, coords

