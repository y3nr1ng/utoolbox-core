import logging
import os

import coloredlogs
import imageio
from tqdm import tqdm

from utoolbox.container.datastore import ImageFolderDatastore
from utoolbox.exposure import histogram
from utoolbox.utils.logging import TqdmLoggingHandler

###
# region: Configure logging facilities
###

logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())

logging.getLogger("tifffile").setLevel(logging.ERROR)

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

###
# endregion
###

data_dir = "data"

imds = ImageFolderDatastore(os.path.join(data_dir, "raw"), imageio.volread)

for im in imds:
    h = histogram(im, 256)
    print(h)
