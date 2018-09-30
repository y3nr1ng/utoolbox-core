import logging
import os

import coloredlogs
import imageio

from utoolbox.smlm.thunderstorm import ThunderSTORM

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####


##### EXCEUTE THUNDERSTORM #####

# ThunderSTORM(ndim, cal_file=None, tmp_dir=None)
#worker = ThunderSTORM(2, tmp_dir='/home2/scratch')
worker = ThunderSTORM(2)
worker.run('src_dir', 'dst_dir')

##### RESULT #####
