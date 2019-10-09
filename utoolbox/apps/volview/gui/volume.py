import logging

from vispy.scene.visuals import Volume

from utoolbox.apps.volview.gui.canvas import Canvas

__all__ = ["VolumeCanvas"]

logger = logging.getLogger(__name__)


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
