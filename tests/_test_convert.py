from pprint import pprint

import coloredlogs
from imageio import imread, volwrite

from utoolbox.container.datastore import (
    ImageDatastore, 
    SparseImageDatastore
)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

with SparseImageDatastore(
    'umanager_mock_dataset', imread, pattern='*561*'
) as ds:
    ImageDatastore.convert_from('umanager_dense_dataset', ds, imageio.volwrite)
    