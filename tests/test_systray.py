import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QApplication, QMenu, QSystemTrayIcon, QWidget, qApp

class SystemTray(QSystemTrayIcon):
    def __init__(self, parent):
        super(SystemTray, self).__init__(parent)
        self.setIcon(QIcon("disconnect.png"))

        info_action = QAction("Hello world!", self)
        info_action.setEnabled(False)

        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(qApp.quit)

        context_menu = QMenu()
        context_menu.addAction(info_action)
        context_menu.addSeparator()
        context_menu.addAction(quit_action)

        self.setContextMenu(context_menu)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = QWidget()
    systray = SystemTray(w)
    systray.show()

    sys.exit(app.exec_())
