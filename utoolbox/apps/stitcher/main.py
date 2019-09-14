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
        self._preview = Preview()

        # default view
        self.setCentralWidget(self._preview)


if __name__ == "__main__":
    from pprint import pprint

    from utoolbox.data.datastore import ImageFolderDatastore

    imfds = ImageFolderDatastore("/scratch/20170606_ExM_cell7", pattern="cell7*")
    pprint(list(imfds.values()))

    raise RuntimeError("DEBUG")

    app = QApplication()
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
