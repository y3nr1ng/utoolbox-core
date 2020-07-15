import glob
import logging
import os

import coloredlogs
import dask.array as da
import imageio
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
import zarr

from utoolbox.util.dask import get_client, batch_submit

logger = logging.getLogger("dask_batch_binning")

# we know this is annoying, silence it
logging.getLogger("tifffile").setLevel(logging.ERROR)

# convert verbose level
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


def main(src_dir):
    src_dir = os.path.abspath(src_dir)
    tif_file_list = glob.glob(os.path.join(src_dir, "*.tif"))
    logger.info(f"found {len(tif_file_list)} file(s) to process")

    # test read
    array = imageio.volread(tif_file_list[0])
    shape, dtype = array.shape, array.dtype
    del array
    logger.info(f"array info {shape}, {dtype}")

    # coordinate list
    csv_file_list = glob.glob(os.path.join(src_dir, "*.csv"))
    columns = {
        "grid_x": int,
        "grid_y": int,
        "grid_z": int,
        "coord_x": float,
        "coord_y": float,
        "coord_z": float,
    }
    coords = pd.read_csv(
        csv_file_list[0],
        skiprows=6,
        usecols=list(range(3, 9)),
        names=columns.keys(),
        dtype=columns,
    )

    @delayed
    def volread_np(uri):
        return np.array(imageio.volread(uri))

    def volread_da(uri):
        return da.from_delayed(volread_np(uri), shape, dtype)

    subsets = []
    for src_path, coord in zip(tif_file_list, coords.itertuples(index=False)):
        array = volread_da(src_path)
        coord = coord._asdict()

        array = xr.DataArray(
            array,
            name="raw",
            dims=["z", "y", "x"],
            coords={k: v for k, v in coord.items()},
        )

        # attach tile coordinate
        array = array.expand_dims("tile")
        array = array.assign_coords({k: ("tile", [v]) for k, v in coord.items()})

        # convert to datasets
        subset = array.to_dataset()
        subsets.append(subset)
    dataset = xr.combine_nested(subsets, concat_dim="tile")

    print(dataset)

    """
    compressor = zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.blosc.SHUFFLE)
    dataset.to_zarr(
        "_demo_converted.zarr", encoding={"raw": {"compressor": compressor}}
    )
    """

    dataset["mip_xy"] = dataset["raw"].max("z")

    mip_dataset = dataset["mip_xy"][dataset.grid_z == 0]

    tasks = []
    counter = 1
    for iy, image_xy in mip_dataset.groupby("grid_y"):
        for ix, image in image_xy.groupby("grid_x"):
            image = image.squeeze()

            fname = f"tile{counter:03d}_x{ix:03d}_y{iy:03d}.tif"
            counter += 1

            tasks.append((fname, image))

    def imwrite(uri, image):
        imageio.imwrite(uri, image)
        print(uri)

    fname, image = zip(*tasks)
    batch_submit(imwrite, fname, image)


if __name__ == "__main__":
    with get_client(address="localhost:8786"):
        main("utoolbox-core/workspace/data/20200704_kidney_demo-2_CamA")
