import logging
from pprint import pprint

import coloredlogs
import pyopencl as cl

import utoolbox.parallel

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

ctx = utoolbox.parallel.create_some_context(dev_type='gpu', vendor='NVIDIA')
print(ctx)
