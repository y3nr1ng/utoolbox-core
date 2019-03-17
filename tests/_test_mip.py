import logging
import os

import coloredlogs
import imageio
import numpy as np
import tqdm

from utoolbox.util.decorator import timeit
from utoolbox.transform.projections import Orthogonal

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


path = "test_ortho.tif"
I = imageio.volread(path)
logger.info("I.shape={}".format(I.shape))


with Orthogonal(I) as ortho:
    Ixy = ortho.xy
    imageio.imwrite("mip_xy.tif", Ixy)
    
    Ixz = ortho.xz
    imageio.imwrite("mip_xz.tif", Ixz)

    Iyz = ortho.yz
    imageio.imwrite("mip_yz.tif", Iyz)