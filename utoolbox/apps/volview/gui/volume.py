from abc import abstractmethod
import logging

from PySide2.QtCore import QObject, Signal
from vispy.scene import SceneCanvas
from vispy.scene.cameras import ArcballCamera
from vispy.scene.visuals import Volume

__all__ = ["VolumeCanvas"]

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


class VolumeCanvas(Canvas):
    def __init__(self, method="translucent", cmap="viridis"):
        logger.debug("VolumeCanvas.__init__")
        super().__init__()
        self.unfreeze()

        self._method = method
        self._cmap = cmap

        self.volume = None

        self.freeze()

    ##

    def on_model_changed(self):
        if self.volume is None:
            logger.debug("create new volume visual")
            # create volume visual
            self.volume = Volume(
                self.model.data,
                method=self._method,
                cmap=self._cmap,
                emulate_texture=False,
            )
            # create viewbox
            viewbox = self.grid.add_view(row=0, col=0)
            viewbox.add(self.volume)
            # attach camera
            viewbox.camera = self.camera
        else:
            logger.debug("update data in the visual")
            self.volume.set_data(self.model.data)

    ##

    @property
    def cmap(self):
        return self._cmap

    @property
    def method(self):
        return self._method
