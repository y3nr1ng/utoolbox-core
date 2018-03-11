import numpy as np
from utoolbox.container import Volume

path = "data/RFiSHp2aLFCYC/decon/488/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif"

raw = Volume(path, resolution=(0.3, 0.102, 0.102))
#raw = Volume(path)
print("raw.shape={}, raw.dtype={}".format(raw.shape, raw.dtype))
print("raw.resolution={}".format(raw.metadata.resolution))

raw_dup = Volume(raw)
print("raw_dup.shape={}, raw_dup.dtype={}".format(raw_dup.shape, raw_dup.dtype))
print("raw_dup.resolution={}".format(raw_dup.metadata.resolution))
