import logging
import os

import coloredlogs
import imageio
import numpy as np

from utoolbox.container import ImageFolderDatastore

##
# region: Setup logger
##
logger = logging.getLogger(__name__)

logging.getLogger("tifffile").setLevel(logging.ERROR)

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
##
# endregion
##

root = (
    "/Users/Andy/Documents/Sinica (Data)/Projects/ExM SIM/20181224_Expan_Tub_Tiling_SIM"
)

ds = ImageFolderDatastore(root, imageio.volread, pattern="RAWcell1_*")
n_phases = 5

root = os.path.join(root, "widefield")
try:
    os.mkdir(root)
except:
    pass
for fn, I in ds.items():
    logger.info('merging "{}"'.format(fn))
    nz, _, _ = I.shape
    J = []
    for iz in range(0, nz, n_phases):
        j = np.sum(I[iz : iz + 5, ...], axis=0, dtype=I.dtype)
        J.append(j)

    new_path = os.path.join(root, fn)
    imageio.volwrite(new_path, np.stack(J, axis=0))
