import logging
import os

from utoolbox.data.dataset.base import MultiChannelDataset

__all__ = ["BigDataViewerDataset"]

logger = logging.getLogger(__name__)


class BigDataViewerDataset(MultiChannelDataset):
    """
    Args:
        root (str): path to the XML file
    """
    def __init__(self, root):
        pass
