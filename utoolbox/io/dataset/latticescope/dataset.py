import glob
import logging
import os

from dask import delayed
import dask.array as da
import imageio

from ..base import DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset

__all__ = ["LatticeScopeDataset", "LatticeScopeTiledDataset"]

logger = logging.getLogger(__name__)


class LatticeScopeDataset(DenseDataset, MultiChannelDataset, MultiViewDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir

        super().__init__()

        self.preload()

    ##

    @property
    def read_func(self):
        def func(uri, shape, dtype):
            array = da.from_delayed(
                delayed(imageio.volread, pure=True)(uri), shape, dtype
            )
            return array

        return func

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def settings_path(self):
        return self._settings_path

    ##

    def _can_read(self):
        # find common prefix
        file_list = glob.glob(os.path.join(self.root_dir, "*.tif"))
        prefix = os.path.commonprefix(file_list)

        # strip until ends with an underscore
        if prefix[-1] != "_":
            prefix = prefix.rsplit("_", 1)[0]

        # find settings
        settings_path = f"{prefix}_Settings.txt"
        if not os.path.exists(settings_path):
            return False

        self._settings_path = settings_path
        return True

    def _enumerate_files(self):
        search_path = os.path.join(self.root_dir, "*.tif")
        return glob.glob(search_path)

    def _load_array_info(self):
        pass

    def _load_channel_info(self):
        pass

    def _retrieve_file_list(self, coord_dict):
        pass

    def _load_metadata(self):
        # TODO parse Settings.txt
        pass

    def _load_view_info(self):
        pass


class LatticeScopeTiledDataset(LatticeScopeDataset, TiledDataset):
    @property
    def script_path(self):
        return self._script_path

    ##

    def _can_read(self):
        if not super()._can_read():
            return False

        # find script file
        script_path = glob.glob(os.path.join(self.root_dir, "*.csv"))

        if len(script_path) == 0:
            return False
        elif len(script_path) > 1:
            script_path = script_path[0]
            logger.warning(
                f'found multiple script file candidates, using "{script_path}"'
            )

        self._script_path = script_path
        return True

    def _load_tiling_coordinates(self):
        pass

    def _load_tiling_info(self):
        pass
        # TODO ln 1-2, summary
        # TODO ln 3-N, position list
