from pprint import pprint

import coloredlogs
import imageio
import numpy as np

from utoolbox.container.datastore import SparseStackImageDatastore

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

with SparseStackImageDatastore(
    'GH146ACV_power100_60ms_z3_split', 
    imageio.imread, 
    pattern='*488nm*'
) as ds:
    for i, im in enumerate(ds):
        avg, std = np.mean(im), np.std(im)
        print("[{:03d}] {:.4f} +/- {:.4f}".format(i, avg, std))