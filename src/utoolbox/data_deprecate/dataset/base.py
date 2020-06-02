from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import logging
import os

from utoolbox.util import AttrDict

logger = logging.getLogger(__name__)

__all__ = ["Dataset", "DatasetInfo", "MultiChannelDataset"]


class DatasetInfo(AttrDict):
    """
    Book-keeping relevant information of the dataset.
    """

    TileInfo = namedtuple("TileInfo", ["index", "extent"])

    def __init__(self):
        """Populate default values."""
        # time
        self.frames = 1

        # color
        self.channels = []

        # stack, 2D
        self.shape = None
        self.pixel_size = None

        # stack, 3D
        self.n_slices = 1
        self.z_step = 0

        # tiles
        self.tiles = []

    ##

    @property
    def tile_shape(self):
        if not self.is_tiled:
            raise RuntimeError("not a tiled dataset")

        indices = [tile.index for tile in self.tiles]
        ax_range = [[None, None]] * len(self.tiles[0].index)
        for index in indices:
            for iax, ax in enumerate(index):
                try:
                    if ax < ax_range[iax][0]:
                        ax_range[iax][0] = ax
                    elif ax > ax_range[iax][1]:
                        ax_range[iax][1] = ax
                except TypeError:
                    ax_range[iax] = [ax, ax]
        return tuple(ax[1] - ax[0] + 1 for ax in ax_range)

    ##

    @property
    def is_tiled(self):
        return len(self.tiles) > 0

    @property
    def is_timeseries(self):
        return self.frames > 1


class Dataset(Mapping, metaclass=ABCMeta):
    """
    Dataset base class.

    Arg:
        root (str): source of the dataset

    Attributes:
        info (DatasetInfo): dataset description parsed from the metadata
        metadata : raw dataset metadata
        root (str): absolute path to the source
    """

    def __init__(self, root):
        self._root = os.path.abspath(os.path.expanduser(root))

        self._metadata, self._info = self._load_metadata(), DatasetInfo()
        self._deserialize_info_from_metadata()

        self._datastore = self._load_datastore()

    def __getitem__(self, key):
        return self._datastore[key]

    def __iter__(self):
        return self._datastore

    def __len__(self):
        return len(self._datastore)

    ##

    @property
    def info(self):
        return self._info

    @property
    def metadata(self):
        """Extracted metadata of this dataset."""
        return self._metadata

    @property
    def root(self):
        """Root of the dataset, need not to be a file object."""
        return self._root

    ##

    def _load_metadata(self):
        pass

    @abstractmethod
    def _deserialize_info_from_metadata(self):
        """Load dataset info."""
        raise NotImplementedError

    @abstractmethod
    def _load_datastore(self):
        """Load actual data as datastore object."""
        raise NotImplementedError


class MultiChannelDataset(Dataset):
    """
    Dataset with multi-color channels.

    Arg:
        root (str): source of the dataset
    """

    def __init__(self, root):
        super().__init__(root)

    def __iter__(self):
        return iter(self._datastore)

    ##

    def _load_datastore(self):
        """Override for multi-channel setup."""
        channels = self.info.channels
        logger.info("{} channel(s)".format(len(channels)))
        return {channel: self._load_channel(channel) for channel in channels}

    @abstractmethod
    def _load_channel(self, channel):
        return NotImplementedError
