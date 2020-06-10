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


def prompt_pixel_size():
    from prompt_toolkit.shortcuts import input_dialog

    value = input_dialog(
        title="Invalid pixel size", text="Please provide the effective pixel size (um):"
    ).run()
    dx = float(value)

    return dx


class MicroManagerV1Dataset(
    DirectoryDataset, DenseDataset, MultiChannelDataset, TiledDataset
):
    @property
    def metadata_path(self):
        return self._metadata_path

    @property
    def read_func(self):
        def func(uri, shape, dtype):
            # layered volume
            nz, shape = shape[0], shape[1:]
            array = da.stack(
                [
                    da.from_delayed(delayed(imageio.imread)(file_path), shape, dtype)
                    for file_path in uri
                ]
            )
            if array.shape[0] != nz:
                logger.warning(
                    f"retrieved layer mis-matched (require: {nz}, provide: {array.shape[0]})"
                )
            return array

        return func

    ##

    def _can_read(self):
        version = self.metadata["summary"]["MicroManagerVersion"]
        return version.startswith("1.")

    def _enumerate_files(self):
        search_path = os.path.join(self.root_dir, "*", "*.tif")
        return glob.glob(search_path)

    def _load_array_info(self):
        # shape
        shape = self.metadata["summary"]["Height"], self.metadata["summary"]["Width"]
        if any(s <= 0 for s in shape):
            logger.warning(f"invalid image size {shape}, using frame metadata")
            shape = self.metadata["frame"]["Height"], self.metadata["frame"]["Width"]
            if any(s <= 0 for s in shape):
                raise MalformedMetadataError("unable to determine correct image shape")

        nz = self.metadata["summary"]["Slices"]
        if nz > 1:
            shape = (nz,) + shape

        # dtype
        ptype = self.metadata["summary"]["PixelType"].lower()
        dtype = {"gray16": np.uint16}[ptype]

        return shape, dtype

    def _load_channel_info(self):
        channels = self.metadata["summary"]["ChNames"]
        if len(channels) == 1 and channels[0] == "Default":
            channels = ["*"]
        return channels

    def _load_metadata(self, filename="metadata.txt"):
        # find all `metadata.txt` and try to open until success
        search_path = os.path.join(self.root_dir, "*", filename)
        for metadata_path in glob.iglob(search_path):
            try:
                with open(metadata_path, "r") as fd:
                    raw_metadata = json.load(fd)
                    logger.info(f'found metadata at "{metadata_path}"')

                    # cleanup metadata
                    metadata = {}
                    metadata["summary"] = raw_metadata["Summary"]
                    for key, value in raw_metadata.items():
                        if (
                            key.startswith("Metadata") and key.endswith(".tif")
                        ) or key.startswith("FrameKey-"):
                            metadata["frame"] = value
                            break
                    else:
                        raise MalformedMetadataError("unable to find frame metadata")
                    self._metadata_path = metadata_path
                    return metadata
            except KeyError:
                pass
            except json.decoder.JSONDecodeError as err:
                logger.warning(f'metadata corruption, reason "{str(err)}"')  # try next
                try:
                    return self._load_corrupted_metadata(metadata_path)
                except ImportError:
                    # failed to use other loader, reraise
                    raise err
        else:
            raise MissingMetadataError()

    def _load_corrupted_metadata(self, path):
        """
        Load corrupted metadata.

        If we identified the metadata, but it is corrupted somewhere, we should already 
        have its file path. 

        Args:
            path (str): path to the metadata
        """
        try:
            import ijson
        except ImportError:
            logger.error('requires "ijson" to perform iterative load')  # unable to try
            raise

        logger.info("attempting iterative load to bypass corruptions")
        metadata = {}
        with open(path, "r") as fd:
            # 1) parse root level keys
            parser, frame_metadata_key = ijson.parse(fd), None
            for prefix, event, value in parser:
                if prefix.startswith("Metadata") and prefix.endswith(".tif"):
                    frame_metadata_key = prefix
                    break
            else:
                raise MalformedMetadataError("unable to find frame metadata")
            # 2) dump
            fd.seek(0)
            metadata["summary"] = next(iter(ijson.items(fd, "Summary")))
            fd.seek(0)
            metadata["frame"] = next(iter(ijson.items(fd, frame_metadata_key)))
        return metadata

    def _parse_position_list(self):
        positions = self.metadata["summary"]["InitialPositionList"]

        frame = self.metadata["frame"]
        xy_stage, z_stage = frame["Core-XYStage"], frame["Core-Focus"]
        logger.debug(f'XY stage is "{xy_stage}", Z stage is "{z_stage}"')

        coords, index = [], []
        labels = []
        for position in positions:
            # NOTE MicroManager only tiles in 2D
            index.append(
                {"x": position["GridColumnIndex"], "y": position["GridRowIndex"]}
            )

            # coordinate
            coord_dict = position["DeviceCoordinatesUm"]
            coord = {}
            try:
                coord_x, coord_y = tuple(coord_dict[xy_stage])
                coord["x"] = float(coord_x)
                coord["y"] = float(coord_y)
            except KeyError:
                pass
            try:
                coord_z = coord_dict[z_stage][0]
                coord["z"] = float(coord_z)
            except KeyError:
                pass
            coords.append(coord)

            labels.append(position["Label"])

        return coords, index, labels

    def _load_mapped_coordinates(self):
        coords, index, labels = self._parse_position_list()

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
        dx, r = (
            self.metadata["summary"]["PixelSize_um"],
            self.metadata["summary"]["PixelAspect"],
        )
        if dx == 0:
            dx = prompt_pixel_size()
        size = (r * dx, dx)

        if self.metadata["summary"]["Slices"] > 1:
            size = (abs(self.metadata["summary"]["z-step_um"]),) + size

        return size

    # def _missing_data(self):
    #    shape, dtype = self._load_array_info()
    #    return delayed(np.zeros)(shape, dtype)

    def _retrieve_file_list(self, coord_dict):
        # find folder that contains the stack
        tile_index = ["tile_x", "tile_y"]
        label = self.tile_coords.xs(
            itemgetter(*tile_index)(coord_dict), axis="index", level=tile_index
        )["label"].iloc[0]
        tile_folder = os.path.join(self.root_dir, label)
        file_list = [f for f in self.files if f.startswith(tile_folder)]

        # find images
        ch = coord_dict["channel"]
        if ch != "*":
            # we need to match channels explicitly
            file_list = [f for f in file_list if ch in f]

        return file_list


class MicroManagerV2Dataset(MicroManagerV1Dataset):
    def _can_read(self):
        version = self.metadata["summary"]["MicroManagerVersion"]
        return version.startswith("2.")

    def _load_channel_info(self):
        channels = super()._load_channel_info()

        # internal bookkeeping
        self._channel_order = channels

        return channels

    def _parse_position_list(self):
        coords, index = [], []
        labels = []
        for position in self.metadata["summary"]["StagePositions"]:
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
                    coord["z"] = float(coord_z)
                else:
                    logger.warning(f"unknown device {device_name}")
            coords.append(coord)

            labels.append(position["Label"])

        return coords, index, labels

    def _load_voxel_size(self):
        frame_metadata = self.metadata["frame"]
        matrix = frame_metadata["PixelSizeAffine"]

        # type cast
        matrix = [float(m) for m in matrix.split(";")]

        # fix affine matrix, default should be identity
        if all(m == 0 for m in matrix):
            dx = prompt_pixel_size()
            matrix[0] = matrix[4] = dx

        binning = frame_metadata["Binning"]
        binning = int(binning)

        # affine matrix
        #   [ 1.0, 0.0, 0.0; 0.0, 1.0, 0.0 ]
        # NOTE
        #   - we don't care about directions at this stage
        #   - MM already factor in binning to the matrix
        size = (abs(matrix[4]), abs(matrix[0]))
        logger.info(f"binning {binning}x, effective pixel size {size} um")

        if self.metadata["summary"]["Slices"] > 1:
            size = (abs(self.metadata["summary"]["z-step_um"]),) + size

        # ensure everything is in `float`
        size = tuple(float(s) for s in size)

        return size

    def _retrieve_file_list(self, coord_dict):
        # find folder that contains the stack
        tile_index = ["tile_x", "tile_y"]
        label = self.tile_coords.xs(
            itemgetter(*tile_index)(coord_dict), axis="index", level=tile_index
        )["label"].iloc[0]
        tile_folder = os.path.join(self.root_dir, label)
        file_list = [f for f in self.files if f.startswith(tile_folder)]

        # find images
        i_ch = self._channel_order.index(coord_dict["channel"])
        i_ch = f"_channel{i_ch:03d}_"
        file_list = [f for f in file_list if i_ch in f]

        return file_list
