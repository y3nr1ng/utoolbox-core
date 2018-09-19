import logging
from pprint import pprint

import coloredlogs

import utoolbox.latticescope as llsm

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### LOAD FILE #####
path = 'test_filenames'
ds = llsm.Dataset(path)


##### DUMP INVENTORY #####
pprint(ds.settings)
