import boltons.debugutils
#boltons.debugutils.pdb_on_exception()

import numpy as np

from utoolbox.io import TimeSeries
import utoolbox.io.primitives as dtype
from utoolbox.utils.decorators import timeit

#ts = TimeSeries(dtype.SimpleVolume, folder='data')
#print('{} time points'.format(len(ts)))

file_path = 'data/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif'

@timeit
def load_file():
    return dtype.SimpleVolume(file_path)

data = load_file()
renderer = VolumeRender(title='demo', data=data)
