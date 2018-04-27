import os
import logging

import numpy as np

import utoolbox.utils.files as fileutils
from utoolbox.container import Raster
from utoolbox.container.layouts import Volume

#####

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = logging.getLogger(__name__)

#####

source_folder = os.path.join(*["data", "20171201_RFiSHp2aLFCYC", "decon", "488"])
file_list = fileutils.list_files(
    source_folder,
    name_filters=[
        fileutils.ExtensionFilter('tif'),
        fileutils.SPIMFilter(channel=0)
    ]
)

print("[0] = {}".format(file_list[0]))

#####

im1 = Raster(file_list[0], layout=Volume)
im1_dup = Raster(im1, layout=Volume)
im2 = Raster(shape=(128, 128))
print(type(im1))
print(im1._layout)
print(type(im1_dup))
print(im1_dup._layout)
print(type(im2))
print(im2._layout)
