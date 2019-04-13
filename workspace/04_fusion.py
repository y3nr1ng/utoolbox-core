import logging
import os

import coloredlogs
import imageio 
import numpy as np

from utoolbox.container.datastore import (
    ImageDatastore
)
from utoolbox.stitching import Sandbox
from utoolbox.stitching.fusion import rmlp2

##
# region: Logging setup
##
logger = logging.getLogger(__name__)

logging.getLogger('tifffile').setLevel(logging.ERROR)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
##
# endregion
##

#ds = ImageDatastore('data/xy', imageio.imread)

#sb = Sandbox(ds)

dst_root = 'data/result'
try:
    os.mkdir(dst_root)
except:
    pass

ds = ImageDatastore('data/raw', imageio.volread)

fn_list = [fn for fn in ds.keys()]

# using first two as demo
I = [ds[fn] for fn in fn_list[:2]]

# process through layers 
nz, _, _ = I[0].shape
for iz in range(nz):
    print("process {}".format(iz))
    Iz = [_I[iz, ...].astype(np.float32) for _I in I] 
   
    Rz = rmlp2(Iz, T=1/255., r=3, K=16, sigma=1)

    imageio.imwrite(
        os.path.join(dst_root, "R_z{}.tif".format(iz)), 
        Rz
    )
