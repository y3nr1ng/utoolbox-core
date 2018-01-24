import boltons.debugutils
#boltons.debugutils.pdb_on_exception()

import numpy as np
import yt
yt.toggle_interactivity()

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
#renderer = VolumeRender(title='demo', data=data)

units = yt.units.unit_registry.UnitRegistry()
print(units.keys())

# associate a field
data = dict(density=(data, 'au'))
bbox = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
ds = yt.load_uniform_grid(data, data['density'][0].shape, length_unit="Mpc", bbox=bbox, nprocs=64)

sc = yt.create_scene(ds)
sc.show()
#sc.save('rendering.png')
