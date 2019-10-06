import logging

from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import QAction, QMainWindow

from utoolbox.apps.volview.gui.volume import VolumeCanvas
from utoolbox.apps.volview.model import TransformModel

__all__ = ["MainWindow"]

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, size=(512, 512)):
        super().__init__()

        self.setWindowTitle("Demo")

        # startup screen size
        self.resize(*size)

        self._setup_menubar()
        self._setup_canvas()

        # use stylesheet to setup dark mode
        # self.setStyleSheet("background-color:black; color:white;")

    def set_model(self, data):
        # TODO bind data to model
        pass

    ##

    def _setup_menubar(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("File")
        # File > Open
        open_action = file_menu.addAction("Open...")
        open_action.setShortcut("Ctrl+O")
        file_menu.addSeparator()
        # File > Exit
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        # Edit
        edit_menu = menubar.addMenu("Edit")
        # Edit > Display as
        display_as_menu = edit_menu.addMenu("Display as")
        # Edit > Display as > Volume
        volume_action = display_as_menu.addAction("Volume")
        # Edit > Display as > Orthoslice
        ortho_action = display_as_menu.addAction("Orthoslice")
        ortho_action.setDisabled(True)
        edit_menu.addSeparator()
        # Edit > Transfer function
        transfer_action = edit_menu.addAction("Transfer function")
        # (seperator)
        edit_menu.addSeparator()
        # Edit > Show coordinate system
        show_axis_action = edit_menu.addAction("Show axis")
        show_axis_action.setCheckable(True)
        # Edit > Show bounding box
        show_box_action = edit_menu.addAction("Show bounding box")
        show_box_action.setCheckable(True)

        # View
        view_menu = menubar.addMenu("View")
        # View > Reset view
        reset_view_action = view_menu.addAction("Reset view")
        # View > Set View
        set_view_menu = view_menu.addMenu("Set view")
        # View > Set View > +XY
        set_xy_pos_action = set_view_menu.addAction("+ XY")
        set_xz_pos_action = set_view_menu.addAction("+ XZ")
        set_yz_pos_action = set_view_menu.addAction("+ YZ")
        # (separator)
        view_menu.addSeparator()
        # View > Take snapshot
        snapshot_action = view_menu.addAction("Take snapshot")

    def _setup_canvas(self):
        self.canvas = VolumeCanvas()
        self.setCentralWidget(self.canvas.native)
        self.tranfsorm = TransformModel()

