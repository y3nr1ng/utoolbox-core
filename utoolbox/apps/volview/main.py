import logging
import sys

from PySide2.QtWidgets import QApplication

from utoolbox.apps.volview.gui import MainWidget
from utoolbox.apps.volview.model import DataModel

__all__ = ["volview"]

logger = logging.getLogger(__name__)


def volview(data, cmap="gray", show=True):
    # TODO datastore/ndarray

    model = DataModel(data)

    # create application
    app = QApplication()
    # create actual user interface
    widget = MainWidget()
    widget.set_model(model)
    widget.show()
    # run the main Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    import imageio

    data = imageio.volread("demo_brain-vessel.tif")
    volview(data)
