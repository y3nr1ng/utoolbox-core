
import coloredlogs
import imageio
import numpy as np

from utoolbox.data.datastore import ImageDatastore

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

## scan valid files ##
ds = ImageDatastore(
    'data/brain_slice/gray',
    read_func=imageio.imread,
    pattern="*"
)

## probe data size 
im = next(iter(ds.values()))
ny, nx = im.shape
nz = len(ds)

## load data
IF = np.empty((nz, ny, nx), np.float32)
for i, im in enumerate(ds.values()):
    IF[i, ...] = im

print(IF.mean())