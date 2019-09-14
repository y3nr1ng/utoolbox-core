import logging
import sys

from PySide2.QtWidgets import QApplication, QMainWindow

from utoolbox.apps.stitcher.gui import Preview
from utoolbox.stitching import Sandbox

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, datastore):
        super().__init__()
        self.setWindowTitle("Stitcher")

        # load the datastore to sandbox
        self.sandbox = Sandbox(datastore)
        self._preview = Preview(self.sandbox)

        # TODO link algorithm with sandbox

        # default view
        self.setCentralWidget(self._preview)


if __name__ == "__main__":
    from pprint import pprint

    import coloredlogs
    import imageio

    from utoolbox.data.datastore import ImageFolderDatastore

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG",
        fmt="%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    imfds = ImageFolderDatastore(
        "/scratch/20170606_ExM_cell7", pattern="cell7*", read_func=imageio.volread
    )

    app = QApplication()
    mw = MainWindow(imfds)
    mw.show()
    sys.exit(app.exec_())
