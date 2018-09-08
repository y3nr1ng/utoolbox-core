from PyQt5.QtWidgets import QMainWindow

from utoolbox.apps.enqueuer.frames import JobsListFrame

class EnqueuerMainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent=parent)
        self._set_ui_design()

    def _set_ui_design(self):
        self.setWindowTitle("Enqueuer")
        self.setFixedSize(800, 400)

        self.central_widget = JobsListFrame()
        self.setCentralWidget(self.central_widget)

        #self.jobslist_table = JobsListTable(self)

if __name__ == '__main__':
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = EnqueuerMainWindow()
    w.show()
    sys.exit(app.exec_())
