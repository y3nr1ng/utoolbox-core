import logging
import os

import coloredlogs
from dask.distributed import Client, LocalCluster
import imageio

from utoolbox.io import open_dataset


def preview_mip(src_ds, dst_dir):
    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass

    z, i = 0, 0
    for _, tile_xy in src_ds.groupby("tile_z"):
        i = 0
        for _, tile_x in tile_xy.groupby("tile_y"):
            for _, tile in tile_x.groupby("tile_x"):
                # print(type(tile))
                uuid = tile.values[0]
                print(uuid)
                data = src_ds[uuid]

                # mip
                mip = data.max(axis=0)

                # retrieve
                mip = mip.compute()
                imageio.imwrite(os.path.join(dst_dir, f"z{z}_t{i:03d}.tif"), mip)

                i += 1
        z += 1


def preview_midplane(src_ds, dst_dir):
    z = src_ds.index.get_level_values("tile_z").unique().values
    mid_z = z[len(z) // 2]
    logger.info(f"mid z plane @ {mid_z}")

    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass

    mid_tiles = src_ds.iloc[src_ds.index.get_level_values("tile_z") == mid_z]
    for cam, tile_xy in mid_tiles.groupby("view"):
        i = 0
        for _, tile_x in tile_xy.groupby("tile_y"):
            for _, tile in tile_x.groupby("tile_x"):
                # print(type(tile))
                uuid = tile.values[0]
                print(uuid)
                data = src_ds[uuid]

                # mip
                nz = data.shape[0]
                data = data[nz // 2, ...]

                # retrieve
                data = data.compute()
                imageio.imwrite(os.path.join(dst_dir, f"{cam}_tile_{i:04d}.tif"), data)

                i += 1


if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    use_local = True

    if use_local:
        logger.info("using local cluster")
        cluster = LocalCluster(n_workers=4, threads_per_worker=4)
        client = Client(cluster)
    else:
        logger.info("using remote cluster")
        client = Client("10.109.20.6:8786")
    logger.info(client)

    src_ds = open_dataset("Y:/ARod/4F/20200317_No5_CamA")
    print(src_ds.inventory)

    logger.info(f"tile by {src_ds.tile_shape}")

    # INPUT (x, y, z) -> TRUE (z, x, y)
    src_ds.remap_tiling_axes({"x": "z", "y": "x", "z": "y"})
    src_ds.flip_tiling_axes(["x", "y"])

    print(src_ds.inventory)

    # import ipdb; ipdb.set_trace()

    dst_dir = "Y:/ARod/4F/20200317_No5_CamA"
    preview_mip(src_ds, dst_dir)

    client.close()
