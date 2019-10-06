import logging

import numpy as np
from PySide2.QtCore import QObject, Signal
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Volume

__all__ = ["VolumeCanvas"]

logger = logging.getLogger(__name__)


class VolumeCanvas(SceneCanvas, QObject):
    model_changed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(keys="interactive")
        self.unfreeze()

        self._model = None

        # add camera
        self.view = self.central_widget.add_view()
        self.view.camera = "arcball"

        # add visual
        dummy = np.ones(shape=(256, 128, 64), dtype=np.uint8)
        self.volume = Volume(
            dummy, method="translucent", emulate_texture=False, parent=self.view.scene
        )

        self.model_changed.connect(self.on_model_changed)

        self.freeze()

    ##

    def on_model_changed(self):
        self.volume.set_data(self.model)

    ##

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if (self.model is None) or (self.model != model):
            self._model = model
            self.model_changed.emit()
