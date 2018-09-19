import logging
import os

import coloredlogs
import imageio
import numpy as np

import utoolbox.utils.files as fileutils
from utoolbox.transform import deskew

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####
#path = os.path.join("data", "sample1_zp6um_561.tif")
path = "result.tif"
I_in = imageio.volread(path)

spacing = (0.102, 0.5)


##### EXCEUTE DESKEW #####
ctx = utoolbox.parallel.create_some_context(dev_type='gpu', vendor='NVIDIA')
with DeskewTransform(ctx, spacing, 32.8, rotate=True) as transform:
    I_out = transform(I_in)


##### RESULT #####
imageio.volwrite("result_deskew.tif", I_out)
