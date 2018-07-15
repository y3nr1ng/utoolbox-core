import logging
import os

import boltons.debugutils
import numpy as np

###
### configure logger
###
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = logging.getLogger(__name__)


boltons.debugutils.pdb_on_exception()


from utoolbox.io.spim import open_dataset
ds = open_dataset("data/inventory", rescan=True)

print(ds)

raise RuntimeError
