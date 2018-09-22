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
path = "/Volumes/Data/Shared/Andy/live_localization/09202018_HalotagEGFPc3Nup153_Hela_Blinking/cell2_CP550_Nup153_Higher_zp4um_50ms_a6p1s_r1p4s"
#path = "test_filenames"
ds = llsm.Dataset(path, refactor=True)


##### DUMP INVENTORY #####
pprint(ds.settings)
