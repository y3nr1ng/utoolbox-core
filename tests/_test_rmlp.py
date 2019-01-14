import glob
import logging
import os

import coloredlogs
import imageio
import numpy as np

from utoolbox.stitching.fusion_rmlp import rmlp

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


src_dir = "../data/fusion/full"
src_files = glob.glob(os.path.join(src_dir, "*"))

# iterate over files
ds = [imageio.volread(path).astype(np.float32) for path in src_files]

R = rmlp(ds, T=1/255., r=4)