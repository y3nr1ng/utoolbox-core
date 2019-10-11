from abc import abstractmethod
import logging
from uuid import uuid4

import numpy as np
from PySide2.QtCore import QObject, Signal
from vispy.scene.visuals import Volume

__all__ = ["SimpleVolumeModel"]

logger = logging.getLogger(__name__)


class Model(QObject):
    data_changed = Signal()

    def __init__(self, data):
        self._id = uuid4().hex
        self._data, self._visual = None, None

        super().__init__()

        self.data_changed.connect(self.update_visual)

        # force visual updates
        self.data = data

    ##

    def update_visual(self):
        if self.visual is None:
            self._create_visual()
        else:
            self.visual.set_data(self.data)

    ##

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self.data_changed.emit()

    @property
    def id(self) -> str:
        return self._id

    @property
    def visual(self):
        return self._visual

    ##

    @abstractmethod
    def _create_visual(self):
        raise NotImplementedError()


class SimpleDataModel(Model):
    def __init__(self, data, **kwargs):
        if not isinstance(data, np.ndarray):
            raise TypeError("data has to be a Numpy array")

        super().__init__(data)

    ##

    @property
    def ndim(self):
        return self._data.ndim


class SimpleVolumeModel(SimpleDataModel):
    def __init__(self, data, **kwargs):
        super().__init__(data)

    ##

    def _create_visual(self):
        logger.info("create new volume visual")
        return Volume(self.data, emulate_texture=False)
