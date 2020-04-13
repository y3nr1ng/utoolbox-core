import logging

from utoolbox.data.dataset.base import MultiChannelDataset

__all__ = ["ImarisDataset"]

logger = logging.getLogger(__name__)


class ImarisDataset(MultiChannelDataset):
    """
    Representation of an Imaris dataset.

    Args:
        path (str): path to the IMS file
    """

    def __init__(self, path):
        pass

    def _load_metadata(self):
        pass

    def _find_channels(self):
        pass

    def _load_channel(self):
        pass
