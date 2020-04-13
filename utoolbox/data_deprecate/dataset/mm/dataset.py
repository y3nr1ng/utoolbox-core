import json
import logging
import os

import imageio

from utoolbox.data.datastore import SparseVolumeDatastore, SparseTiledVolumeDatastore
from utoolbox.data.dataset.base import DatasetInfo, MultiChannelDataset
from .error import MetadataError, NoMetadataInTileFolderError, NoSummarySectionError

__all__ = ["MicroManagerV1Dataset", "MicroManagerV2Dataset"]

logger = logging.getLogger(__name__)


class MicroManagerV1Dataset(MultiChannelDataset):
    """
    Representation of Micro-Manager V1 dataset stored in sparse stack format.

    Args:
        root (str): source directory of the dataset
        merge (bool, optional): return merged result for tiled dataset
        force_stack (bool, optional): force the dataset to be interpreted as stacks
    """

    def __init__(self, root, merge=True, force_stack=False):
        if not os.path.exists(root):
            raise FileNotFoundError("invalid dataset root")
        if merge and force_stack:
            logger.warning("force output as stack, merging request is ignored")
        self._merge, self._force_stack = merge, force_stack
        super().__init__(root)

    def _load_metadata(self):
        meta_dir = self.root
        # select the first folder that contains `metadata.txt`
        for _path in os.listdir(self.root):
            _path = os.path.join(meta_dir, _path)
            if os.path.isdir(_path):
                meta_dir = _path
                break
        if meta_dir == self.root:
            raise NoMetadataInTileFolderError()
        self._root = meta_dir
        logger.debug('using metadata from "{}"'.format(self.root))
        meta_path = os.path.join(self.root, "metadata.txt")

        try:
            with open(meta_path, "r") as fd:
                # discard frame specific info
                return json.load(fd)
        except KeyError:
            raise NoSummarySectionError()

    def _deserialize_info_from_metadata(self):
        info, summary = self.info, self.metadata["Summary"]

        ver_str = summary["MicroManagerVersion"]
        if ver_str.startswith("1."):
            self._parser()
        else:
            raise MetadataError("not v1 metadata")

    def _parser(self):
        info, summary = self.info, self.metadata["Summary"]

        # time
        info.frames = summary["Frames"]

        # color
        info.channels = summary["ChNames"]

        # stack, 2D
        info.shape = (summary["Height"], summary["Width"])
        dx, r = summary["PixelSize_um"], summary["PixelAspect"]
        if dx == 0:
            logger.warning("pixel size undefined, default to 1")
            dx = 1
        info.pixel_size = (r * dx, dx)

        # stack, 3D
        info.n_slices = summary["Slices"]
        info.z_step = abs(summary["z-step_um"])

        # deserialize position extents
        if summary["Positions"] > 1:
            # tiled dataset
            grids = summary["InitialPositionList"]
            for grid in grids:
                # index
                index = (grid["GridRowIndex"], grid["GridColumnIndex"])

                # extent
                extent_xy, extent_z = (0, 0), None
                for key, value in grid["DeviceCoordinatesUm"].items():
                    if "XY" in key:
                        extent_xy = tuple(value[::-1])
                    elif "Z" in key:
                        extent_z = (value[0],)
                if extent_z is not None:
                    extent = extent_z + extent_xy
                else:
                    extent = extent_xy

                # save
                info.tiles.append(DatasetInfo.TileInfo(index=index, extent=extent))
            logger.info(f"dataset is a {info.tile_shape} grid")

            # extract prefix across folders
            prefix = os.path.commonprefix([grid["Label"] for grid in grids])
            i = prefix.rfind("_")
            if i > 0:
                prefix = prefix[:i]
            logger.debug('folder prefix "{}"'.format(prefix))
            self._folder_prefix = prefix

            # reset root folder one level up, we are one level down in one of the tile
            self._root, _ = os.path.split(self.root)

    def _find_channels(self):
        return self.info.channels

    def _load_channel(self, channel):
        kwargs = {
            "read_func": imageio.imread,
            "folder_pattern": "{}*".format(self._folder_prefix),
            "file_pattern": "*_{}_*".format(channel),
            "extensions": ["tif"],
        }
        if self.info.is_tiled and not self._force_stack:
            return SparseTiledVolumeDatastore(
                self.root,
                tile_shape=self.info.tile_shape,
                tile_order="F",
                merge=self._merge,
                **kwargs,
            )
        else:
            return SparseVolumeDatastore(self.root, sub_dir=False, **kwargs)


class MicroManagerV2Dataset(MicroManagerV1Dataset):
    """
    Representation of Micro-Manager V2 dataset stored in sparse stack format.
    """

    def _deserialize_info_from_metadata(self):
        info, summary = self.info, self.metadata["Summary"]

        ver_str = summary["MicroManagerVersion"]
        if ver_str.startswith("2."):
            self._parser()
        else:
            raise MetadataError("not v2 metadata")

    def _parser(self):
        info, summary = self.info, self.metadata["Summary"]
        sample_frame = None
        for key in self.metadata.keys():
            if key.startswith("Metadata"):
                sample_frame = self.metadata[key]
                break
        else:
            raise RuntimeError(
                "malformed metadata format, unable to find sample frame info"
            )

        # time
        info.frames = summary["Frames"]

        # color
        info.channels = summary["ChNames"]
        # fix default color name
        if len(info.channels) == 1:
            if info.channels[0] == "Default":
                # wildcard, select every file in the folder
                info.channels[0] = "*"

        # stack, 2D
        info.shape = (sample_frame["Height"], sample_frame["Width"])
        dx, matrix = sample_frame["PixelSizeUm"], sample_frame["PixelSizeAffine"]
        # calculate affine matrix
        #   [ 1.0, 0.0, 0.0; 0.0, 1.0, 0.0 ]
        matrix = [float(m) for m in matrix.split(";")]
        info.pixel_size = (matrix[4] * dx + matrix[5], matrix[0] * dx + matrix[2])

        # stack, 3D
        info.n_slices = summary["Slices"]
        info.z_step = abs(summary["z-step_um"])

        # deserialize position extents
        if summary["Positions"] > 1:
            # tiled dataset
            grids = summary["StagePositions"]
            for grid in grids:
                # index
                index = (grid["GridRow"], grid["GridCol"])

                # extent
                extent = []
                for label in ("DefaultZStage", "DefaultXYStage"):
                    label = grid[label]
                    if label:
                        # sanity check before extracting positions
                        for device in grid["DevicePositions"]:
                            if device["Device"] == label:
                                extent += device["Position_um"][::-1]
                                break
                        else:
                            raise RuntimeError(
                                f"stage {label} in use, but position info is missing"
                            )
                extent = tuple(extent)

                # save
                info.tiles.append(DatasetInfo.TileInfo(index=index, extent=extent))
            logger.info(f"dataset is a {info.tile_shape} grid")

            # extract prefix across folders
            prefix = os.path.commonprefix([grid["Label"] for grid in grids])
            i = prefix.rfind("_")
            if i > 0:
                prefix = prefix[:i]
            logger.debug('folder prefix "{}"'.format(prefix))
            self._folder_prefix = prefix

            # reset root folder one level up, we are one level down in one of the tile
            self._root, _ = os.path.split(self.root)

    def _load_channel(self, channel):
        index = self.info.channels.index(channel)
        return super()._load_channel(f"channel{index:03d}")

