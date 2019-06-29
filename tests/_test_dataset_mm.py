from pprint import pprint

import coloredlogs
import imageio
import numpy as np

# import objgraph

from utoolbox.data.dataset import MicroManagerDataset
from utoolbox.data.datastore import VolumeTilesDatastore

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s  %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


class DummyImage(object):
    def __init__(self, path, shape=(32, 32), dtype=np.uint16):
        self._path = path
        self._data = np.empty(shape, dtype)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def path(self):
        return self._path

    @property
    def shape(self):
        return self._data.shape


def dummy_read(path):
    return DummyImage(path)


ds = MicroManagerDataset("perfuse_lectin594_poststain_lectin647_5")

for name, im in ds["640_10X"].items():
    pprint(im)
