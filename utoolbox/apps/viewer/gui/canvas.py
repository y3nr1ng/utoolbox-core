from abc import abstractmethod
import logging

from PySide2.QtCore import QObject, Signal
from vispy.scene import SceneCanvas
from vispy.scene.cameras import ArcballCamera

__all__ = ["Canvas"]

logger = logging.getLogger(__name__)


class Canvas(QObject, SceneCanvas):
    model_changed = Signal()

    def __init__(self):
        # NOTE somehow, super() failed to __init__ both parent class
        # super().__init__()
        QObject.__init__(self)
        SceneCanvas.__init__(self)
        self.unfreeze()

        # model
        self._model = None

        # view
        self._grid = self.central_widget.add_grid()
        self.camera = ArcballCamera()

        # signal
        self.model_changed.connect(self.on_model_changed)

        self.freeze()

    ##

    @abstractmethod
    def on_model_changed(self):
        raise NotImplementedError()

    ##

    @property
    def grid(self):
        return self._grid

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.model_changed.emit()
