import logging

import coloredlogs
from dask.distributed import Client, LocalCluster

from utoolbox.io.dataset import BigDataViewerDataset, LatticeScopeTiledDataset

if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    cluster = LocalCluster(n_workers=4, threads_per_worker=4)
    client = Client(cluster)
    # client = Client("10.109.20.6:8786")
    logger.info(client)

    src_ds = LatticeScopeTiledDataset("X:/ARod/4F/20191212_4F/flybrain_1")
    print(src_ds.inventory)

    dst_dir = "U:/Andy/20191212_4F/flybrain_1_bdv_vds_6"
    BigDataViewerDataset.dump(
        dst_dir,
        src_ds,
        pyramid=[(1, 1, 1), (1, 4, 4)],
        compression=None,
        client=client,
        dry_run=True,
    )

    client.close()
