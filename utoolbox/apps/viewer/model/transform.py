import logging

from PySide2.QtCore import QObject

__all__ = ["TransformModel"]

logger = logging.getLogger(__name__)


class TransformModel(QObject):
    def __init__(self):
        super().__init__()
    