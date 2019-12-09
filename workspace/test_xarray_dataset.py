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

    print("== loaded")

    from dask.distributed import Client
    client = Client()
    print(client)
    print()

    from dask.cache import Cache
    cache = Cache(2e9)
    cache.register() 
    
    from dask.diagnostics import ProgressBar

    # OK
    # coord = {
    #    "channel": "488",
    #    "tile_x": 1919.6500244140625,
    #    "tile_y": -488.0260009765625,
    # }
    # tile = ds[coord]
    # print(tile)
    # tile = ds.query(tile_x=3370.7395, tile_y=-488.026, channel="488")#[:, ::4, ::4]
    # tile = ds.dataset.sel(tile_x=1919.65, tile_y=-488.026)["488"][:, ::4, ::4]
    # ERROR
    # tile = ds.dataset.sel(tile_x=-2433.619, tile_y=-488.026)["488"][:, ::4, ::4]
    # print(".. selected")
    # print(tile)
    # print()

    dst_dir = "_debug"
    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass

    import imageio

    for j, (_, ds_x) in enumerate(ds.groupby("tile_y")):
        for i, (_, tile) in enumerate(ds_x.groupby("tile_x")):
            print(f".. iter (i: {i}, j: {j})")

            uuid = tile.values[0]
            with ProgressBar():
                data = ds[uuid][:, ::4, ::4].max(axis=0).compute()
                print(f"{data.shape}, {data.dtype}")
                dst_path = os.path.join(dst_dir, f"tile-{i:03d}-{j:03d}_mip.tif")
                imageio.imwrite(dst_path, data)
