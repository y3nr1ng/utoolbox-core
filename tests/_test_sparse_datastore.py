from pprint import pprint

import coloredlogs

from utoolbox.container.datastore import SparseImageDatastore

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

dummy_read = lambda x: x
with SparseImageDatastore(
    'umanager_mock_dataset', dummy_read, pattern='*561*'
) as ds:
    pprint(ds.files)