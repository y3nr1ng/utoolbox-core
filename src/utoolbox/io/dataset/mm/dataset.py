import glob
import json
import logging
import os
from collections import defaultdict
from operator import itemgetter

import dask.array as da
import imageio
import numpy as np
import pandas as pd
from dask import delayed

from ..base import DenseDataset, DirectoryDataset, MultiChannelDataset, TiledDataset
from .error import MalformedMetadataError, MissingMetadataError

__all__ = ["MicroManagerV1Dataset", "MicroManagerV2Dataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class MicroManagerV1Dataset(
    DirectoryDataset, DenseDataset, MultiChannelDataset, TiledDataset
):
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

    def _load_mapped_coordinates(self):  # TODO update to new format
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

    def _load_mapped_coordinates(self):
        coords, index = [], []
        labels = []
        for position in self.metadata["StagePositions"]:
            # NOTE MicroManager only tiles in 2D
            index.append({"x": position["GridCol"], "y": position["GridRow"]})

            xy_stage, z_stage = position["DefaultXYStage"], position["DefaultZStage"]
            coord = {}
            for device in position["DevicePositions"]:
                device_name = device["Device"]
                if device_name == xy_stage:
                    coord_x, coord_y = tuple(device["Position_um"])
                    coord["x"] = float(coord_x)
                    coord["y"] = float(coord_y)
                elif device_name == z_stage:
                    coord_z = device["Position_um"][0]
                    coords["z"] = float(coord_z)
                else:
                    logger.warning(f"unknown device {device_name}")
            coords.append(coord)

            labels.append(position["Label"])

        coords, index = pd.DataFrame(coords), pd.DataFrame(index)
        labels = pd.DataFrame({"label": labels})

        # rename index
        index_names_mapping = {ax: f"tile_{ax}" for ax in index.columns}
        index.rename(index_names_mapping, axis="columns", inplace=True)

        # sanity check
        if len(coords) != len(index) or len(coords) != len(labels):
            raise MalformedMetadataError("coordinate info incomplete")

        # build multi-index
        df = pd.concat([index, coords, labels], axis="columns")
        df.set_index(index.columns.to_list(), inplace=True)

        # rename coords
        coord_names_mapping = {ax: f"{ax}_coord" for ax in coords.columns}
        df.rename(coord_names_mapping, axis="columns", inplace=True)

        return df

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
        # find folder that contains the stack
        tile_index = ["tile_x", "tile_y"]
        label = self.tile_coords.xs(
            itemgetter(*tile_index)(coord_dict), axis="index", level=tile_index,
        )["label"].iloc[0]
        tile_folder = os.path.join(self.root_dir, label)
        file_list = [f for f in self.files if f.startswith(tile_folder)]

        # find images
        i_ch = self._channel_order.index(coord_dict["channel"])
        i_ch = f"_channel{i_ch:03d}_"
        file_list = [f for f in file_list if i_ch in f]

        return file_list
