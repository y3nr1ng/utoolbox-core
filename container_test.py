import logging
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = logging.getLogger(__name__)

import numpy as np
from utoolbox.container import Volume

path = "data/20171201_RFiSHp2aLFCYC/decon/488/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif"

raw = Volume(path, resolution=(0.3, 0.102, 0.102))
#raw = Volume(path)
print("raw.shape={}, raw.dtype={}".format(raw.shape, raw.dtype))
print("raw.resolution={}".format(raw.metadata.resolution))

raw_dup = Volume(raw)
print("raw_dup.shape={}, raw_dup.dtype={}".format(raw_dup.shape, raw_dup.dtype))
print("raw_dup.resolution={}".format(raw_dup.metadata.resolution))

print(raw_dup.metadata)

print("raw.ndim={}".format(raw.ndim))

raw_xy = np.amax(raw, axis=0)
print("raw_xy.ndim={}".format(raw_xy.ndim))
print("raw_xy.resolution={}".format(raw_xy.metadata.resolution))
