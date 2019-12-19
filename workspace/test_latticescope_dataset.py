import logging

import coloredlogs
from dask.distributed import Client, LocalCluster

from utoolbox.io.dataset import BDVDataset, LatticeScopeTiledDataset

if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    logger.info(client)

    src_ds = LatticeScopeTiledDataset("X:/ARod/20191212_4F/flybrain_1")
    print(src_ds.inventory)

    dst_dir = "U:/Andy/20191212_4F/flybrain_1_bdv"
    BDVDataset.dump(dst_dir, src_ds, pyramid=[(1, 1, 1), (2, 4, 4)], client=client)
