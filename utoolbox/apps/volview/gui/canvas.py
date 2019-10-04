import logging

import numpy as np
from PySide2.QtCore import Signal
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Volume

__all__ = ["Canvas"]

logger = logging.getLogger(__name__)


class Canvas(SceneCanvas):
    model_changed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._model = None

        # initialize the view
        view = self.central_widget.add_view()

        # add volume visual, use dummy
        dummy = np.zeros(shape=(16, 16, 16), dtype=np.uint16)
        volume = Volume(
            dummy,
            method="translucent",
            parent=view.scene,
            emulate_texture=False,
        )
        self.volume = volume

        self.model_changed.connect(self.on_model_changed)

        self.refresh()

    ##

    def on_draw(self):
        gloo.clear(color="block", depth=True)
        self.volume.draw()

    def on_resize(self, event):
        # set viewport and reconfigure visual transforms
        viewport = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*viewport)
        self.volume.transform.configure(canvas=self, viewport=viewport)

    ##

    def refresh(self):
        pass

    def on_model_changed(self):
        self.volume.set_data(self.model)
        self.refresh()

    ##

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if (self.model is None) or (self.model != model):
            self._model = model
            self.model_changed.emit()
