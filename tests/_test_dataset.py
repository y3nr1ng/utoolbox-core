import logging
from pprint import pprint

import coloredlogs
import pytest

import utoolbox.latticescope as llsm

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### LOAD FILE #####
path = "mock_dataset"
ds = llsm.Dataset(path, refactor=False)


##### DUMP INVENTORY #####
pprint(ds.settings)

pprint(ds.datastore)
for k, v in ds.datastore.items():
    print(" << {} >>".format(k))
    pprint(v.files)
