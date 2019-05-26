from pprint import pprint

import coloredlogs
from utoolbox.container.datastore import FolderCollectionDatastore

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s  %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

ds = FolderCollectionDatastore("sparse", read_func=lambda x: x)
for k, v in ds.items():
    print("[ {} ]".format(k))
    pprint(v)

