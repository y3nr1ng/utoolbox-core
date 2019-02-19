from pprint import pprint

import coloredlogs
import imageio

from utoolbox.container.datastore import SparseImageDatastore

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

with SparseImageDatastore(
    'GH146ACV_power100_60ms_z3_split', 
    imageio.imread, 
    pattern='*488nm*'
) as ds:
    for im in ds.read():
        pprint(ds.files)