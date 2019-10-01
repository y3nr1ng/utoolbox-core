import logging

from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import QWidget

from utoolbox.apps.volview.gui import Canvas
from utoolbox.apps.volview.model import TransformModel

__all__ = ["MainWidget"]

logger = logging.getLogger(__name__)


class MainWidget(QWidget):
    def __init__(self, size=None):
        super().__init__()

        # DEBUG init size
        if not size:
            size = (1024, 1024)
        self._setup_init_size(size)

        self._setup_canvas()

        # use stylesheet for dark mode
        self.setStyleSheet("background-color:black; color:white;")

    def set_model(self, data):
        # TODO bind data to model
        pass

    ##

    def _setup_init_size(self, size):
        self.resize(*size)

    def _setup_canvas(self):
        self.canvas = Canvas()
        self.tranfsorm = TransformModel()

