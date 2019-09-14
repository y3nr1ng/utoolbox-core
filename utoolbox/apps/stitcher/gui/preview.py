from collections import OrderedDict
import logging

from PySide2.QtWidgets import QWidget, QGridLayout, QPushButton
import vispy.app
from vispy.visuals import ImageVisual
from vispy.visuals.transforms import STTransform

__all__ = ["Preview"]

logger = logging.getLogger(__name__)


class Preview(QWidget):
    def __init__(self, sandbox):
        super().__init__()

        self._setup_canvas()
        self._images = OrderedDict()

        self._init_from_sandbox(sandbox)

    ##

    def add_image(self, data, name=None, position=(0, 0)):
        if name is None:
            name = f"untitled_{len(self._images)}"

        image = ImageVisual(data, method="subdivide")

        # DEBUG
        # scale and center image in canvas
        s = 700.0 / max(data.shape)
        t = 0.5 * (700.0 - (data.shape[0] * s)) + 50
        image.transform = STTransform(scale=(s, s), translate=(t, 50))

    ##

    def _init_from_sandbox(self, sandbox):
        """
        Load all the images in a datastore and spread them out.
        """
        raise RuntimeError("DEBUG")

    def _setup_canvas(self):
        self.setLayout(QGridLayout(self))
        self.canvas = vispy.app.Canvas()
        self.layout().addWidget(self.canvas.native)
