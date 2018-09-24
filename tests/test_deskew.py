import logging
import os

import coloredlogs
import imageio
import numpy as np
import pycuda.driver as cuda

from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform import DeskewTransform

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####
path = "deskew_input.tif"
I_in = imageio.volread(path)
logger.info("I_in.shape={}".format(I_in.shape))

spacing = (0.102, 0.5)


##### EXCEUTE DESKEW #####
ctx = create_some_context()
ctx.push()

transform = DeskewTransform(spacing, 32.8, rotate=True)
I_out = transform(I_in)

cuda.Context.pop()


##### RESULT #####
imageio.volwrite("deskew_output.tif", I_out)
