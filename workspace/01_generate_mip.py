import logging
import os

import coloredlogs
import imageio
from tqdm import tqdm

from utoolbox.container.datastore import ImageDatastore
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
imds = ImageDatastore(
    os.path.join(data_dir, 'raw'),
    lambda x: (x, imageio.volread(x))
)

logger.info("create output directories")
projs = ('xy', 'xz', 'yz')
for view in projs:
    try:
        os.makedirs(os.path.join(data_dir, view))
    except:
        pass

for fp, I in tqdm(imds):
    with Orthogonal(I) as ortho:
        for view in projs:
            J = getattr(ortho, view)

            fn = os.path.basename(fp)
            fp = os.path.join(data_dir, view, fn)
            imageio.imwrite(fp, J)