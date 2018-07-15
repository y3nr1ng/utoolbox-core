import logging
import os

import boltons.debugutils
import imageio
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

import utoolbox.io.imageio
store = utoolbox.io.imageio.ImageioDataStore("data/deskew_sample2.tif")

from xarray import Dataset, backends, conventions
from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
ds = conventions.decode_cf(
            store, mask_and_scale=False, decode_times=False,
            concat_characters=False, decode_coords=False,
            drop_variables=None)

if len(ds.data_vars) != 1:
    raise ValueError('Given file dataset contains more than one data '
                     'variable. Please read with xarray.open_dataset and '
                     'then select the variable you want.')
else:
    da, = ds.data_vars.values()

da._file_obj = ds._file_obj

# Reset names if they were changed during saving
# to ensure that we can 'roundtrip' perfectly
if DATAARRAY_NAME in ds.attrs:
    da.name = ds.attrs[DATAARRAY_NAME]
    del ds.attrs[DATAARRAY_NAME]

if da.name == DATAARRAY_VARIABLE:
    da.name = None

print(da)

raise RuntimeError
