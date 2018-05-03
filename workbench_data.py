import logging
import os

import imageio
import numpy as np

import utoolbox.utils.files as fileutils
from utoolbox.container import Raster
from utoolbox.container.layouts import Volume
from utoolbox.transform import deskew
from utoolbox.utils.decorators import timeit

#####

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = logging.getLogger(__name__)

#####

source_folder = os.path.join(*["data", "20171201_RFiSHp2aLFCYC", "raw", "488"])
file_list = fileutils.list_files(
    source_folder,
    name_filters=[
        fileutils.ExtensionFilter('tif'),
        fileutils.SPIMFilter(channel=0)
    ]
)

print("[0] = {}".format(file_list[0]))

#####

im1 = Raster(file_list[0], layout=Volume, spacing=(.5, .102, .102))
logger.debug(im1)
logger.debug("im1.layout={}".format(im1.metadata.layout))

#####

@timeit
def operation():
    return deskew(im1, 30)
im2 = operation()
imageio.volwrite("data/test.tif", im2)
