import boltons.debugutils
boltons.debugutils.pdb_on_exception()

import numpy as np

from utoolbox.utils.decorators import timeit

file_path = 'data/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif'

@timeit
def load_file():
    import utoolbox.io.primitives as dtype
    return dtype.SimpleVolume(file_path)

data = load_file()
print(data.shape)
print(type(data))
