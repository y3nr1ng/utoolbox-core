import logging
from math import hypot
import os
from pprint import pprint

import coloredlogs
import imageio
import numpy as np
#from tqdm import tqdm

from skimage.feature import register_translation

from utoolbox.container.datastore import ImageDatastore
from utoolbox.stitching import Sandbox
#from utoolbox.feature import DftRegister
#from utoolbox.util.logging import TqdmLoggingHandler

from utoolbox.util.decorator import timeit

###
# region: Configure logging facilities
###
logger = logging.getLogger(__name__)
#logger.addHandler(TqdmLoggingHandler())

logging.getLogger('tifffile').setLevel(logging.ERROR)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
###
# endregion
###

def tuple_float_to_str(t, digits=4):
    float_fmt = '{{:.{}f}}'.format(digits)
    fmt = '(' + ', '.join([float_fmt, ] * len(t)) + ')'
    return fmt.format(*list(t))

data_dir = 'data'
projs = ('xy', 'xz', 'yz')

ds = ImageDatastore(
    os.path.join(data_dir, projs[0]),
    imageio.imread
)

# use arbitrary item as size template
im = next(iter(ds.values()))
logger.info("shape={}, {}".format(im.shape, im.dtype))

# upsampling factor
uf = 10
# overlap ratio
overlap_perct = 0.2

# minimum area = 
ny, nx = im.shape
min_overlap_area = (ny*overlap_perct) * (nx*overlap_perct)

keys = [key for key in ds.keys()]
sb = Sandbox(ds)

n_im = len(ds)
for i_ref in range(n_im):
    adj_list = []
    for i_tar in range(i_ref+1, n_im):
        shifts, error, _ = register_translation(
            ds[keys[i_ref]], ds[keys[i_tar]], upsample_factor=uf
        )

        # filter out-of-range shifts
        dy, dx = shifts
        ny, nx = im.shape
        overlap_area = (ny-abs(dy)) * (nx-abs(dx))

        if (overlap_area < 0) or (overlap_area < min_overlap_area):
            continue
        sb.link(keys[i_ref], keys[i_tar], shifts, error)

        #adj_list.append([i_tar, error, shifts, hypot(*shifts)])

sb.consolidate()
