from pprint import pprint

import coloredlogs
import imageio
import numpy as np

# import objgraph

from utoolbox.container.dataset import MicroManagerDataset
from utoolbox.container.datastore import VolumeTilesDatastore

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


with VolumeTilesDatastore(
    "perfuse_lectin594_poststain_lectin647_5",
    read_func=dummy_read,
    folder_pattern="1-Pos_*",
    file_pattern="*_561_*",
    tile_shape=(3, 4),
    return_as="plane",
) as ds:
    pprint(ds._uri)
    # pprint(ds.metadata)

    # print("\n=== objgraph ===")
    # objgraph.show_growth(limit=30)
    # print()

# objgraph.show_growth(limit=30)

mmds = MicroManagerDataset("perfuse_lectin594_poststain_lectin647_5")
#pprint(mmds.metadata)