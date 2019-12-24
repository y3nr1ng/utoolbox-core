import logging

import coloredlogs
from dask.distributed import Client, LocalCluster
import imageio
import numpy as np

from utoolbox.io.dataset import BigDataViewerDataset, LatticeScopeTiledDataset

if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    # cluster = LocalCluster(n_workers=4, threads_per_worker=4)
    # client = Client(cluster)
    # client = Client("10.109.20.6:8786")
    # logger.info(client)

    src_ds = LatticeScopeTiledDataset("Y:/ARod/6F/20191223")
    print(src_ds.inventory)

    # import ipdb; ipdb.set_trace()

    z = src_ds.index.get_level_values("tile_z").unique().values
    mid_z = z[len(z) // 2]

    mid_tiles = src_ds.iloc[src_ds.index.get_level_values("tile_z") == mid_z]
    for cam, tile_xy in mid_tiles.groupby("view"):
        i = 0
        for _, tile_x in tile_xy.groupby("tile_y"):
            for _, tile in tile_x.groupby("tile_x"):
                uuid = tile.values[0]
                print(uuid)
                data = src_ds[uuid].compute()
                imageio.imwrite(f"{cam}_tile_{i:04d}.tif", data)
                i += 1

    # dst_dir = "U:/Andy/20191212_4F/flybrain_1_bdv_vds_6"
    # BigDataViewerDataset.dump(
    #    dst_dir,
    #    src_ds,
    #    pyramid=[(1, 1, 1), (1, 4, 4)],
    #    compression=None,
    #    client=client,
    #    dry_run=True,
    # )

    # client.close()
