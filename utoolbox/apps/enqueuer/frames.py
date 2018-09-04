from PyQt5.QtWidgets import (QFrame,
                             QGroupBox,
                             QHBoxLayout, QVBoxLayout,
                             QPushButton, QTableWidget)

class JobsListFrame(QFrame):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent=parent)
        self._set_ui_design()

    def closeEvent(self, event):
        super(JobsListFrame, self).closeEvent(event)
        #TODO close internal widgets here, .clear() then .close()
        self.close()

    def _set_ui_design(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        jobs_group = QGroupBox()
        jobs_group_layout = QHBoxLayout(jobs_group)
        jobs_group_layout.addWidget(QTableWidget())

        priority_group = QGroupBox()
        priority_group_layout = QVBoxLayout(priority_group)
        priority_group_layout.addWidget(QPushButton("+"))
        priority_group_layout.addWidget(QPushButton("-"))
        jobs_group_layout.addWidget(priority_group)

        main_layout.addWidget(jobs_group)

        op_group = QGroupBox()
        op_group_layout = QHBoxLayout(op_group)
        op_group_layout.addWidget(QPushButton("Edit"))
        op_group_layout.addWidget(QPushButton("Run"))
        op_group_layout.addWidget(QPushButton("Exit"))
        main_layout.addWidget(op_group)
