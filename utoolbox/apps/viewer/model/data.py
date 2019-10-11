import logging

from PySide2.QtCore import QObject, Signal

__all__ = ["DatastoreModel", "SimpleDataModel"]

logger = logging.getLogger(__name__)


class Model(QObject):
    data_changed = Signal()

    def __init__(self, prefetch_size=0):
        super().__init__()


class DatastoreModel(QObject):
    def __init__(self, datastore, **kwargs):
        super().__init__(**kwargs)


class SimpleDataModel(Model):
    def __init__(self, data, **kwargs):
        super().__init__()
        self._data = data

    ##

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self.data_changed.emit()
