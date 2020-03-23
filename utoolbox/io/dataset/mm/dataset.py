from collections import defaultdict
import glob
import json
import logging
from operator import itemgetter
import os

from dask import delayed
import dask.array as da
import imageio
import numpy as np
import pandas as pd

from ..base import DenseDataset, MultiChannelDataset, TiledDataset

from .error import MalformedMetadataError, MissingMetadataError

__all__ = ["MicroManagerV1Dataset", "MicroManagerV2Dataset"]

logger = logging.getLogger(__name__)


class MicroManagerV1Dataset(DenseDataset, MultiChannelDataset, TiledDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir

        super().__init__()

        self.preload()

    ##

    @property
    def metadata_path(self):
        return self._metadata_path

    @property
    def read_func(self):
        def func(uri, shape, dtype):
            # order by z
            uri.sort()

            # layered volume
            nz, shape = shape[0], shape[1:]
            array = da.stack(
                [
                    da.from_delayed(delayed(imageio.imread)(file_path), shape, dtype)
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
        if any(s <= 0 for s in shape):
            logger.warning(f"metadata incidcates image size has 0 ({shape}), fixing...")
            with open(self.metadata_path, "r") as fd:
                metadata = json.load(fd)
                for key, block in metadata.items():
                    try:
                        if block["Height"] > 0 and block["Width"] > 0:
                            shape = block["Height"], block["Width"]
                            logger.info(f'using image shape info from "{key}"')
                            break
                    except KeyError:
                        pass
                else:
                    raise MalformedMetadataError(
                        "unable to determine correct image shape"
                    )

        nz = self.metadata["Slices"]
        if nz > 1:
            shape = (nz,) + shape

        # dtype
        ptype = self.metadata["PixelType"].lower()
        dtype = {"gray16": np.uint16}[ptype]

        return shape, dtype

    def _load_channel_info(self):
        return self.metadata["ChNames"]

    def _load_metadata(self, metadata_name="metadata.txt"):
        # find all `metadata.txt` and try to open until success
        search_path = os.path.join(self.root_dir, "*", metadata_name)
        for metadata_path in glob.iglob(search_path):
            try:
                with open(metadata_path, "r") as fd:
                    metadata = json.load(fd)
                    logger.info(f'found metadata at "{metadata_path}"')
                    self._metadata_path = metadata_path
                    return metadata["Summary"]
            except KeyError:
                pass
        else:
            raise MissingMetadataError()

    def _load_tiling_coordinates(self):
        positions = self.metadata["InitialPositionList"]

        coords = defaultdict(list)
        labels = dict()
        for position in positions:
            # coordinate
            coord_dict = position["DeviceCoordinatesUm"]
            try:
                coord_x, coord_y = tuple(coord_dict["XY Stage"])
                coord_x, coord_y = np.float32(coord_x), np.float32(coord_y)

                coords["tile_x"].append(coord_x)
                coords["tile_y"].append(coord_y)
            except KeyError:
                pass
            try:
                coord_z = coord_dict["Z Stage"][0]
                coord_z = np.float32(coord_z)

                coords["tile_z"].append(coord_z)
            except KeyError:
                pass

            # label
            # NOTE MicroManager only tiles in 2D, no need to include Z for indexing
            labels[(coord_x, coord_y)] = position["Label"]

        # internal bookkeeping
        self._tile_prefix = labels

        return pd.DataFrame({k: np.array(v) for k, v in coords.items()})

    def _load_voxel_size(self):
        dx, r = self.metadata["PixelSize_um"], self.metadata["PixelAspect"]
        if dx == 0:
            logger.warning("pixel size undefined, default to 1")
            dx = 1
        size = (r * dx, dx)

        if self.metadata["Slices"] > 1:
            size = (abs(self.metadata["z-step_um"]),) + size

        return size

    # def _missing_data(self):
    #    shape, dtype = self._load_array_info()
    #    return delayed(np.zeros)(shape, dtype)

    def _retrieve_file_list(self, coord_dict):
        prefix = self._tile_prefix[itemgetter("tile_x", "tile_y")(coord_dict)]
        return glob.glob(
            os.path.join(self.root_dir, prefix, f"*_{coord_dict['channel']}_*.tif")
        )


class MicroManagerV2Dataset(MicroManagerV1Dataset):
    def _can_read(self):
        version = self.metadata["MicroManagerVersion"]
        return version.startswith("2.")

    def _load_channel_info(self):
        channels = self.metadata["ChNames"]
        if len(channels) == 1 and channels[0] == "Default":
            channels = ["*"]

        # internal bookkeeping
        self._channel_order = channels

        return channels

    def _load_tiling_coordinates(self):
        positions = self.metadata["StagePositions"]

        coords = defaultdict(list)
        labels = dict()
        for position in positions:
            devices = position["DevicePositions"]
            xy_stage, z_stage = position["DefaultXYStage"], position["DefaultZStage"]
            for device in devices:
                if device["Device"] == xy_stage:
                    coord_y, coord_x = tuple(device["Position_um"])
                    coord_y, coord_x = np.float32(coord_y), np.float32(coord_x)

                    coords["tile_x"].append((coord_x))
                    coords["tile_y"].append((coord_y))
                elif device["Device"] == z_stage:
                    coord_z = device["Position_um"][0]
                    coord_z = np.float32(coord_z)

                    coords["tile_z"].append(coord_z)
                else:
                    logger.warning(f"unknown device {device['Device']}")

            # label
            # NOTE MicroManager only tiles in 2D, no need to include Z for indexing
            labels[(coord_x, coord_y)] = position["Label"]

        # internal bookkeeping
        self._tile_prefix = labels

        return pd.DataFrame({k: np.array(v) for k, v in coords.items()})

    def _load_voxel_size(self):
        # extract sample frame from the metadata file
        sample_frame = None
        with open(self.metadata_path, "r") as fd:
            metadata = json.load(fd)
            for key in metadata.keys():
                if key.startswith("Metadata"):
                    sample_frame = metadata[key]
                    break
            else:
                raise MalformedMetadataError()

        dx, matrix = sample_frame["PixelSizeUm"], sample_frame["PixelSizeAffine"]
        # calculate affine matrix
        #   [ 1.0, 0.0, 0.0; 0.0, 1.0, 0.0 ]
        matrix = [float(m) for m in matrix.split(";")]
        size = (matrix[4] * dx + matrix[5], matrix[0] * dx + matrix[2])

        if self.metadata["Slices"] > 1:
            size = (abs(self.metadata["z-step_um"]),) + size

        return size

    def _retrieve_file_list(self, coord_dict):
        prefix = self._tile_prefix[itemgetter("tile_x", "tile_y")(coord_dict)]
        i_ch = self._channel_order.index(coord_dict["channel"])
        return glob.glob(
            os.path.join(self.root_dir, prefix, f"*_channel{i_ch:03d}_*.tif")
        )
