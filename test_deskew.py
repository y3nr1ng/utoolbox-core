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

"""
source_folder = os.path.join(*["data", "20170112_RFiSHp2aLFCYC", "raw", "488"])
file_list = fileutils.list_files(
    source_folder,
    name_filters=[
        fileutils.ExtensionFilter('tif'),
        fileutils.SPIMFilter(channel=0)
    ]
)

print("[0] = {}".format(file_list[0]))
"""

#file_path = os.path.join("data", "sample1_zp6um_561.tif")
file_path = os.path.join("data", "deskew_sample2.tif")

#####

im1 = Raster(file_path, layout=Volume, spacing=(1., .102, .102))
logger.debug(im1)
logger.debug("im1.layout={}".format(im1.metadata.layout))

#####

@timeit
def operation():
    return deskew(im1, 0.5, rotate=False)
im2 = operation()
logger.debug("type(im2)={}".format(type(im2)))
imageio.volwrite("data/output.tif", im2)
