import logging

from .generic import BaseDataset

__all__ = ["SerializableDataset"]

logger = logging.getLogger(__name__)


class SerializableDataset(BaseDataset):
    def __init__(self):
        pass
        # TODO specify serialization/deserialization functions
