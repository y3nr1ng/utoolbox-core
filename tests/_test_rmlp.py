import glob
import logging
from multiprocessing import Pool
import os

import coloredlogs
import imageio
import numpy as np

from utoolbox.container.datastore import ImageFolderDatastore
from utoolbox.stitching.fusion import rmlp2

logging.getLogger("tifffile").setLevel(logging.ERROR)

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

ds = ImageFolderDatastore(
    "../workspace/data/tls_exm/raw", read_func=imageio.volread, pattern="*_tilingp*"
)

Is = []
for fn, I in ds.items():
    print('reading "{}"'.format(fn))
    Is.append(I)
nz, _, _ = Is[0].shape
logger.info("{} layers".format(nz))

dst_root = "{}_rmlp".format(ds.root)
try:
    os.mkdir(dst_root)
except:
    pass

# process by layers
for iz in range(nz // 2, nz // 2 + 1):
    print("process {}".format(iz))
    Iz = [I[iz, ...].astype(np.float32) for I in Is]

    Iz = [Iz[0], Iz[2], Iz[4]]
    for i, I in enumerate(Iz):
        imageio.imwrite(os.path.join(dst_root, "I_{}_z{}.tif".format(i, iz)), I)

    Rz = rmlp2(Iz, T=1 / 255.0, r=3, K=16, sigma=1)

    imageio.imwrite(os.path.join(dst_root, "R_z{}.tif".format(iz)), Rz)

# pool = Pool(4)
# pool.map(process, range(539, nz))

# R = np.stack(R, axis=0)

