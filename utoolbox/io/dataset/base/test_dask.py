import glob
import os
from pprint import pprint

from dask import delayed
import dask.array as da
import imageio
import xarray as xr

from utoolbox.utils.decorator import timeit

# path = "/scratch/GH146ACV_power100_60ms_z3"
# path = "/scratch/20170718_U2Os_BLStimu/cell2_FTmChG6s_zp6um_20ms_interval_6s/raw"
path = "/scratch/20170606_ExM_cell7"
file_list = glob.glob(os.path.join(path, "*.tif"))
print(file_list[:5])

# simple read
reader = imageio.get_reader(file_list[0])

# TIFF metadata
metadata = reader.get_meta_data()
pprint(metadata)
# number of layers
nz = reader.get_length()
# read one layer
im_tmp = reader.get_next_data()
ny, nx = im_tmp.shape
dtype = im_tmp.dtype

shape = nz, ny, nx
print(f"{shape}, {dtype}")
print()

reader.close()

data = [
    da.from_delayed(delayed(imageio.volread)(file_path), shape, dtype)
    for file_path in file_list
]


@timeit
def dask():
    data0 = xr.DataArray(data[1])
    print(data0.mean().compute())


dask()
print()


@timeit
def raw():
    data = imageio.volread(file_list[1])
    print(data.mean())


raw()
print()
