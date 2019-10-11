import logging
import sys

from PySide2.QtWidgets import QApplication

from utoolbox.apps.viewer.gui import MainWindow
from utoolbox.apps.viewer.model import SimpleVolumeModel

__all__ = ["viewer"]

logger = logging.getLogger(__name__)


def viewer(data, cmap="gray", show=True):
    # TODO datastore/ndarray

    # create application
    app = QApplication()
    main = MainWindow(size=(768, 768))

    # create model
    model = SimpleVolumeModel(data)
    main.add_model(model)

    # show and wait forever in the event loo
    main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import coloredlogs
    import imageio

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    # DEBUG data source, somewhere
    for path in ("/scratch/t1-head-demo.tif", "G:/t1-head-demo.tif"):
        try:
            data = imageio.volread(path)
            break
        except FileNotFoundError:
            pass
    else:
        raise FileNotFoundError("unable to find the test data")
    viewer(data)
