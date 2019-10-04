import json
import logging
import os

import imageio

from utoolbox.data.datastore import SparseVolumeDatastore, SparseTiledVolumeDatastore
from utoolbox.data.dataset.base import DatasetInfo, MultiChannelDataset
from .error import NoMetadataInTileFolderError, NoSummarySectionError

logger = logging.getLogger(__name__)


class MicroManagerDataset(MultiChannelDataset):
    """
    Representation of Micro-Manager dataset stored in sparse stack format.

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
                return json.load(fd)["Summary"]
        except KeyError:
            raise NoSummarySectionError()

    def _deserialize_info_from_metadata(self):
        info = self.info

        # time
        info.frames = self.metadata["Frames"]

        # color
        info.channels = self.metadata["ChNames"]

        # stack, 2D
        info.shape = (self.metadata["Height"], self.metadata["Width"])
        dx, r = self.metadata["PixelSize_um"], self.metadata["PixelAspect"]
        if dx == 0:
            logger.warning("pixel size unset, default to 1")
            dx = 1
        info.pixel_size = (r * dx, dx)

        # stack, 3D
        info.n_slices = self.metadata["Slices"]
        info.z_step = abs(self.metadata["z-step_um"])

        if self.metadata["Positions"] > 1:
            # tiled dataset
            grids = self.metadata["InitialPositionList"]
            for grid in grids:
                # index
                index = (grid["GridRowIndex"], grid["GridColumnIndex"])

                # extent
                extent_xy, extent_z = (0, 0), (0, )
                for key, value in grid["DeviceCoordinatesUm"].items():
                    if "XY" in key:
                        extent_xy = tuple(value[::-1])
                    elif "Z" in key:
                        extent_z = (value[0],)
                extent = extent_z + extent_xy

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

