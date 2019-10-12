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
        self._model = []

        # NOTE somehow, super() failed to __init__ both parent class
        # super().__init__()
        QObject.__init__(self)
        SceneCanvas.__init__(self)
        self.unfreeze()

        # view
        self._grid = self.central_widget.add_grid()
        self.camera = ArcballCamera()

        # signal
        self.model_changed.connect(self.on_model_changed)

        self.freeze()

    def __delitem__(self, key):
        # TODO remove parent from model
        model = self[key]
        self.model.remove(model)

    def __getitem__(self, key):
        try:
            return self.model[key]
        except KeyError:
            # partial match
            for model in self.model:
                if model.id.startswith(key):
                    return model
            else:
                raise KeyError(f"unable to find model {key}")

    def __setitem__(self, key, value):
        self.model.append(value)
        self.model_changed.emit()

    def __iter__(self):
        return self.model

    def __len__(self):
        return len(self.model)

    ##

    @abstractmethod
    def on_model_changed(self):
        # update canvas
        raise NotImplementedError()

    ##

    @property
    def grid(self):
        return self._grid

    @property
    def model(self):
        return self._model
