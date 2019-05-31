from abc import abstractmethod
from collections.abc import Mapping
import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["Dataset", "MultiChannelDataset"]


class Dataset(Mapping):
    """
    Dataset base class.

    Parameters
    ----------
    root : str
        Source of the dataset.
    """

    def __init__(self, root):
        self._root = os.path.abspath(os.path.expanduser(root))
        self._metadata = self._load_metadata()
        self._datastore = self._load_datastore()

    def __getitem__(self, key):
        return self._datastore[key]

    def __iter__(self):
        return self._datastore

    def __len__(self):
        return len(self._datastore)

    @property
    def metadata(self):
        """Extracted metadata of this dataset."""
        return self._metadata

    @property
    def root(self):
        """Root of the dataset, need not to be a file object."""
        return self._root

    def to_hdf(self, dst_root=None, virtual=True):
        raise NotImplementedError

    @abstractmethod
    def _load_datastore(self):
        """Load actual data as datastore object."""
        raise NotImplementedError

    def _load_metadata(self):
        pass


class MultiChannelDataset(Dataset):
    """
    Dataset with multi-color channels.

    Parameters
    ----------
    root : str
        Source of the dataset.
    """

    def __init__(self, root):
        super().__init__(root)

    def __iter__(self):
        return iter(self._datastore)

    @abstractmethod
    def _find_channels(self):
        return NotImplementedError

    @abstractmethod
    def _load_channel(self, channel):
        return NotImplementedError

    def _load_datastore(self):
        """Override for multi-channel setup."""
        channels = self._find_channels()
        logger.info("{} channel(s)".format(len(channels)))
        return {channel: self._load_channel(channel) for channel in channels}
