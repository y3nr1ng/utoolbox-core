import glob
import logging
from multiprocessing import Pool
import os

import coloredlogs
import imageio
import numpy as np

from utoolbox.container.datastore import ImageDatastore
from utoolbox.stitching.fusion import rmlp2

logging.getLogger('tifffile').setLevel(logging.ERROR)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

ds = ImageDatastore(
    #'../data/fusion/crop', 
    '/Users/Andy/Documents/Sinica (Data)/Projects/ExM SIM/20181224_Expan_Tub_Tiling_SIM/widefield',
    #'/Volumes/SSD-2/fuse',
    imageio.volread,
    pattern='RAWcell1_*'
)

Is = []
for fn, I in ds.items():
    print("reading \"{}\"".format(fn))
    Is.append(I)
nz, _, _ = Is[0].shape
logger.info("{} layers".format(nz))

dst_root = 'test'
try:
    os.mkdir(dst_root)
except:
    pass

# process by layers
for iz in range(nz//2, nz//2+1):
    print("process {}".format(iz))
    Iz = [I[iz, ...].astype(np.float32) for I in Is] 
   
    Rz = rmlp2(Iz, T=1/255., r=3, K=16, sigma=1)

    imageio.imwrite(os.path.join(dst_root, "R_z{}.tif".format(iz)), Rz)

#pool = Pool(4)
#pool.map(process, range(539, nz))

#R = np.stack(R, axis=0)


