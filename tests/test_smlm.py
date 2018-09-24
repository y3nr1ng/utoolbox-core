import logging
import os

import coloredlogs
import imageio

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####


##### EXCEUTE DESKEW #####


##### RESULT #####
