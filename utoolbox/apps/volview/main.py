import logging
import sys

from PySide2.QtWidgets import QApplication

from utoolbox.apps.volview.gui import MainWindow
from utoolbox.apps.volview.model import SimpleDataModel

__all__ = ["volview"]

logger = logging.getLogger(__name__)


def volview(data, cmap="gray", show=True):
    # TODO datastore/ndarray

    model = SimpleDataModel(data)

    # create application
    app = QApplication()
    # create actual user interface
    main = MainWindow()
    main.set_model(model)
    main.show()
    # run the main Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    import imageio

    data = imageio.volread("/scratch/t1-head-demo.tif")
    volview(data)
