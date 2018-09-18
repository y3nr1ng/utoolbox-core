"""
Table with integrated progressbar.

References
----------
[1] Dunya-desktop: A modular, customizable and open-source desktop application
    for accessing and visualizing music data. Available at: https://bit.ly/2wFhIz7
"""
class TableView(QTableView):
    def __init__(self, parent=None):
        super(TableView, self).__init__()

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setMouseTracking(True)

        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()

        self.setWordWrap(False)

        self._set_font()

    def _set_font(self, size=12):
        """Sets the font."""
        font = QFont()
        font.setPointSize(size)
        self.setFont(font)

class TableWidget(QTableWidget, TableView):
    def __init__(self, parent=None):
        QTableWidget.__init__(self, parent=parent)
        TableView.__init__(self, parent=parent)

        self._set_column_header()
        self.setDisabled(True)

    def _set_column_header(self):
        pass

    def add_item(self, text):
        self.insertRow(self.rowCount())

        i_row = self.rowCount()-1
        self.set_status(i_row, 0)
        source = QTableWidgetItem(text)
        self.setItem(i_row, 1, source)

    def set_status(self, raw, exist=None):
        item = QLabel()
        item.setAlignment(Qt.AlignCenter)

        #TODO

    def set_progressbar(self, status):
        docid = status.docid
        n_progress = status.n_progress

        self.setCellWidget(self.indices[docid], 0, ProgressBar(self))
        progressbar = self.cellWidget(self.indices[docid], 0)

        if progressbar:
            if not setp == n_progress:
                progressbar.update_progress_bar(step, n_progress)
            else:
                self.set_status(self.indices[docid], 1)
                self.refresh_row(docid)

    def refresh_row(self, docid):
        row = self.indices[docid]
        if self.item(row, 1):
            if setMouseTracking:
                pass

class JobsListTable(TableWidget):
    def __init__(self, parent=None):
        QTableWidget.__init__(self, parent=parent)
        TableView.__init__(self, parent=parent)
