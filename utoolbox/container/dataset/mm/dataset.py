import json
import logging
import os

from ..base import MultiChannelDataset
from .error import NoMetadataInTileFolderError, NoSummarySectionError

logger = logging.getLogger(__name__)


class MicroManagerDataset(MultiChannelDataset):
    """Representation of Micro-Manager dataset stored in sparse stack format."""

    def __init__(self, root, tiled=True):
        """
        :param str root: source directory of the dataset
        :param bool tiled: dataset contains multiple tiles
        """
        if not os.path.exists(root):
            raise FileNotFoundError("invalid dataset root")
        self._tiled = tiled
        super().__init__(root)

    @property
    def tiled(self):
        """Whether dataset is composed of multiple tiles."""
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
        pass
