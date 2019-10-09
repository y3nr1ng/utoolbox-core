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
    main = MainWindow(size=(768, 768))
    main.set_model(model)
    main.show()
    # run the main Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    import coloredlogs
    import imageio

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

<<<<<<< HEAD
    data = imageio.volread("/scratch/t1-head-demo.tif")
=======
    data = imageio.volread("G:/t1-head-demo.tif")
>>>>>>> 9886c19f25c11cf39fc41b0230e3a4a71525c079
    volview(data)
