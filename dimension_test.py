import boltons.debugutils
boltons.debugutils.pdb_on_exception()

import numpy as np

from utoolbox.utils.decorators import timeit

file_path = 'F:\\fusion_raw\\cell7_zp3um_561_1.tif'

@timeit
def load_file():
    import utoolbox.io.primitives as dtype
    return dtype.SimpleVolume(file_path)

data = load_file()
print(data.shape)
