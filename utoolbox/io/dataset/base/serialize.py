import logging

from .generic import GenericDataset

__all__ = ["SerializableDataset"]

logger = logging.getLogger(__name__)


class SerializableDataset(GenericDataset):
    def __init__(self):
        pass
        # TODO specify serialization/deserialization functions
