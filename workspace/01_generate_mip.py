import logging
import os

import coloredlogs
import imageio
from tqdm import tqdm

from utoolbox.container.datastore import ImageFolderDatastore
from utoolbox.transform.projections import Orthogonal
from utoolbox.util.logging import TqdmLoggingHandler

###
# region: Configure logging facilities
###
logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())

logging.getLogger('tifffile').setLevel(logging.ERROR)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
###
# endregion
###

data_dir = 'data'
src_ds = ImageFolderDatastore(
    os.path.join(data_dir, 'raw'),
    read_func=imageio.volread
)

logger.info("create output directories")
projs = ('xy', 'xz', 'yz')

mip_ds = [
    ImageFolderDatastore(os.path.join(
        data_dir, view), 
        write_func=imageio.imwrite
    )
    for view in projs
]

for filename, I in tqdm(src_ds.items()):
    with Orthogonal(I) as ortho:
        for i, view in enumerate(projs):
            J = getattr(ortho, view)
            mip_ds[i][filename] = J