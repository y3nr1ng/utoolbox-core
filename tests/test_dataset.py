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
path = "/Volumes/Data/Shared/Andy/live_localization/09142018_HalotagEGFPc3Nup153+PmeI/clone2_zp4um_100ms_interval_13s"
ds = llsm.Dataset(path, refactor=True)


##### DUMP INVENTORY #####
pprint(ds.settings)
