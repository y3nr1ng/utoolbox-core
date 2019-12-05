import logging
import os
from pprint import pprint

import coloredlogs

from utoolbox.io.dataset.mm import MicroManagerV1Dataset

if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    src_dir = "C:/Users/Andy/Downloads/20191119_ExM_kidney_10XolympusNA06_zp5_7x8_DKO_4-2_Nkcc2_488_slice_2_1"
    ds = MicroManagerV1Dataset(src_dir)

    print("dataset info")
    print(ds.dataset)
    print()

    # from dask.distributed import Client
    # client = Client()
    # print(client)
    # print()

    from dask.diagnostics import ProgressBar

    # OK
    # tile = ds.dataset.sel(tile_x=3370.7395, tile_y=-488.026)["488"][:, ::4, ::4]
    #tile = ds.dataset.sel(tile_x=1919.65, tile_y=-488.026)["488"][:, ::4, ::4]
    # ERROR
    tile = ds.dataset.sel(tile_x=-2433.619, tile_y=-488.026)["488"][:, ::4, ::4]
    print(tile)
    print()

    import imageio

    with ProgressBar():
        data = tile.compute()
        print(f"{data.shape}, {data.dtype}")
        imageio.volwrite("_debug.tif", data.values)
    # print(f"tile (0, 0) mean: {result}")
