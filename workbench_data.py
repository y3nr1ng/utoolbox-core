import os
import logging

import numpy as np

import utoolbox.utils.files as fileutils
from utoolbox.container import Raster
from utoolbox.io.layouts import Volume

#####

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])
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

raw = Raster(Volume, file_list[0], resolution=(0.3, 0.102, 0.102))
print("type(raw) = {}".format(type(raw)))
print("resolution = {}".format(raw.metadata.resolution))

raw_xy = np.amax(raw, axis=0)
print("type(raw_xy) = {}".format(type(raw_xy)))
print("resolution = {}".format(raw_xy.metadata.resolution))
