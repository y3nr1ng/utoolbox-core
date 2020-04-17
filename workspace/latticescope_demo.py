import logging
import os

import coloredlogs

from utoolbox.io.dataset import LatticeScopeTiledDataset


if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="INFO", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    src_ds = LatticeScopeTiledDataset.load("data/demo_3x3x1_CMTKG-V3")
    print(src_ds.inventory)

    logger.info(f"tile by {src_ds.tile_shape}")
