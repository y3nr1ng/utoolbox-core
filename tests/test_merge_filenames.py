import logging

import coloredlogs
import imageio
import numpy as np
import pyopencl as cl

from utoolbox.latticescope import merge_filenames

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


with open("test_filenames.txt", 'r') as fd:
    filenames = fd.readlines()
filenames = [fn.strip() for fn in filenames]

filenames = concat_timestamps(filenames)

with open("test_filenames_merged.txt", 'w') as fd:
    for filename in filenames:
        fd.write("{}\n".format(filename))
