import cupy as cp
from imageio import volread, volwrite
import numpy as np

from utoolbox.filters.perona_malik import PeronaMalik3D
from utoolbox.exposure.rescale_intensity import RescaleIntensity

in_data = volread("data/mito/mito.tif")

ri = RescaleIntensity()

##
# region: PM filter
##
thre = in_data.std()
print("thre={:.4f}".format(thre))
pm = PeronaMalik3D(threshold=thre, niter=16)

in_data = in_data.astype(np.float32)
in_data = cp.asarray(in_data)

out_data = pm(in_data, in_place=True)
out_data = ri(out_data, out_range=cp.uint16)
volwrite("_debug.tif", cp.asnumpy(out_data))
##
# endregion
##
