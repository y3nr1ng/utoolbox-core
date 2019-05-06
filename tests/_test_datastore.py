from functools import partial
import os

import coloredlogs
import imageio 

from utoolbox.container.datastore import (
    ImageDatastore
)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

src_ds = ImageDatastore(
    'GH146ACV_power100_60ms_z3_split_converted', 
    read_func=imageio.volread
)

dst_ds = ImageDatastore(
    'hello_world',
    write_func=imageio.volwrite,
    immutable=False
)

for filename, data in src_ds.items():
    #print(filename)
    # downsample the data
    data = data[..., ::4, ::4]

    base, ext = os.path.splitext(filename)
    filename = "{}_bin4{}".format(base, ext)
    dst_ds[filename] = data