import json
import logging
import os

from ..base import MultiChannelDataset
from .error import NoMetadataInTileFolderError, NoSummarySectionError

logger = logging.getLogger(__name__)


class MicroManagerDataset(MultiChannelDataset):
    """
    Representation of Micro-Manager dataset stored in sparse stack format.

    Args:
        root (str): Source directory of the dataset.
        tiled (bool, optional): Dataset contains multiple tiles.
    """

    def __init__(self, root, tiled=True):
        if not os.path.exists(root):
            raise FileNotFoundError("invalid dataset root")
        self._tiled = tiled
        super().__init__(root)

    @property
    def tiled(self):
        """bool: Dataset contains mulitple tiles."""
        return self._tiled

    def _load_metadata(self):
        # determine root
        path = self.root
        if self.tiled:
            # select the first folder that contains `metadata.txt`
            for _path in os.listdir(self.root):
                _path = os.path.join(path, _path)
                if os.path.isdir(_path):
                    path = _path
                    break
            if path == self.root:
                raise NoMetadataInTileFolderError()
            logger.debug('using metadata from "{}"'.format(path))
        path = os.path.join(path, "metadata.txt")

        with open(path, "r") as fd:
            metadata = json.load(fd)
        # return summary only, discard frame specific info
        try:
            return metadata["Summary"]
        except KeyError:
            raise NoSummarySectionError()

    def _find_channels(self):
        return self.metadata["ChNames"]

    def _load_channel(self, channel):
        logger.debug("_load_channel={}".format(channel))
        raise NotImplementedError()
