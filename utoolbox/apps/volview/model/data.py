import logging

from PySide2.QtCore import QObject

__all__ = ["DataModel"]

logger = logging.getLogger(__name__)


class DataModel(QObject):
    def __init__(self, datastore, prefetch_size=0):
        super().__init__()

