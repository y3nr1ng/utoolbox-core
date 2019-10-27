import logging

from utoolbox.apps.viewer.gui.canvas import Canvas

__all__ = ["VolumeCanvas"]

logger = logging.getLogger(__name__)


class VolumeCanvas(Canvas):
    def __init__(self, method="translucent", cmap="viridis"):
        self._method, self._cmap = method, cmap  # TODO move cmap to model
        self._viewbox = None

        super().__init__()

    ##

    def on_model_changed(self):
        logger.debug("re-assign node parents")
        for model in self.model:
            self.viewbox.add(model.visual)

    ##

    @property
    def cmap(self):
        return self._cmap

    @property
    def method(self):
        return self._method

    @property
    def viewbox(self):
        if self._viewbox is None:
            logger.info("create (0, 0) viewbox")
            self._viewbox = self.grid.add_view(0, 0)
            self._viewbox.camera = self.camera
        return self._viewbox
