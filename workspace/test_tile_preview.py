import logging
import os

import coloredlogs
import dask.array as da
from dask.distributed import Client, LocalCluster, progress
import imageio
import napari
import numpy as np

from utoolbox.io.dataset import LatticeScopeTiledDataset


def load_dataset(path):
    ds = LatticeScopeTiledDataset(path)

    # INPUT (x, y, z) -> TRUE (z, x, y)
    ds.remap_tiling_axes({"x": "z", "y": "x", "z": "y"})
    ds.flip_tiling_axes(["x", "y"])

    print(ds.inventory)

    desc = tuple(f"{k}={v}" for k, v in zip(("x", "y", "z"), reversed(ds.tile_shape)))
    logger.info(f"tiling dimension ({', '.join(desc)})")

    return ds


if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    client = Client("10.109.20.6:8786")
    logger.info(client)

    ds = load_dataset("Y:/ARod/4F/20191227/flybrain")

    # import ipdb; ipdb.set_trace()

    tile_xy = ds.loc[:, "CamA"]
    z = ds.index.get_level_values("tile_z").unique().values

    # index for lookup
    x = ds.index.get_level_values("tile_x").unique().values
    y = ds.index.get_level_values("tile_y").unique().values

    def abs2rel(*args):
        return y.index(args[0]), x.index(args[1])

    # build layered preview
    layers = []
    for z, tile_xy in tile_xy.groupby("tile_z"):
        print(z)
        layer = []
        for y, tile_x in tile_xy.groupby("tile_y"):
            row = []
            for x, tile in tile_x.groupby("tile_x"):
                uuid = tile.values[0]
                print(uuid)

                data = ds[uuid].max(axis=0)

                row.append(data)
            layer.append(row)
        layers.append(layer)
    preview = da.block(layers)
    logger.debug(f"preview.shape={preview.shape}")

    preview = preview.persist()
    progress(preview)
    preview = preview.compute()

    logger.info("save preview")
    imageio.volwrite("preview.tif", preview)

    logger.info("launching napari")
    with napari.gui_qt():
        v = napari.view_image(preview, is_pyramid=False, ndisplay=2)

    client.close()
