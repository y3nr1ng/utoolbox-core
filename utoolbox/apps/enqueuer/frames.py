from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QFrame,
                             QGroupBox,
                             QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QPushButton, QTableWidget, QTableWidgetItem)

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

        jobs_group = QGroupBox()
        jobs_group_layout = QHBoxLayout(jobs_group)
        self.job_list = QTableWidget(4, 2)
        jobs_group_layout.addWidget(self.job_list)

        self.job_list.setHorizontalHeaderLabels(['Path', 'Progress'])
        new_job = QTableWidgetItem("data/demo")
        self.job_list.setItem(0, 0, new_job)

        priority_group = QGroupBox()
        priority_group_layout = QVBoxLayout(priority_group)

        size = QSize(36, 36)
        button = QPushButton('', self)
        #button.clicked.connect(self.handleButton)
        button.setIcon(QIcon('../widgets/resources/up.png'))
        button.setIconSize(size)
        button.setFixedSize(size)
        priority_group_layout.addWidget(button)

        button = QPushButton('', self)
        #button.clicked.connect(self.handleButton)
        button.setIcon(QIcon('../widgets/resources/down.png'))
        button.setIconSize(size)
        button.setFixedSize(size)
        priority_group_layout.addWidget(button)
    
        jobs_group_layout.addWidget(priority_group)

        main_layout.addWidget(jobs_group)

        op_group = QGroupBox()
        op_group_layout = QHBoxLayout(op_group)

        button = QPushButton('Add', self)
        #button.clicked.connect(self.handleButton)
        button.setIcon(QIcon('../widgets/resources/add_item.png'))
        button.setIconSize(QSize(24, 24))
        op_group_layout.addWidget(button)

        op_group_layout.addWidget(QPushButton("Edit"))
        op_group_layout.addWidget(QPushButton("Delete"))
        op_group_layout.addWidget(QPushButton("Run"))
        op_group_layout.addWidget(QPushButton("Exit"))
        main_layout.addWidget(op_group)
