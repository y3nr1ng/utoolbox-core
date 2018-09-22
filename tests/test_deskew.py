import logging
import os

import coloredlogs
import imageio
import numpy as np

from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform import DeskewTransform

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####
path = "../data/sample1_zp6um_561.tif"
I_in = imageio.volread(path)

spacing = (0.102, 0.5)


##### EXCEUTE DESKEW #####
ctx = create_some_context()
transform = DeskewTransform(spacing, 32.8, rotate=True)
I_out = transform(I_in)


##### RESULT #####
#imageio.volwrite("result_deskew.tif", I_out)
