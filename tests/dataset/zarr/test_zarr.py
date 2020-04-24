import logging
import os

from utoolbox.io.dataset import LatticeScopeTiledDataset, ZarrDataset
from utoolbox.io.dataset import TiledDatasetIterator

logger = logging.getLogger("test_zarr")


def main():
    pwd = os.path.abspath(__file__)
    cwd = os.path.dirname(pwd)
    parent = os.path.dirname(cwd)

    logger.info("loading source dataset")
    ds_src_dir = os.path.join(parent, "data", "demo_3D_2x2x2_CMTKG-V3")
    ds_src = LatticeScopeTiledDataset.load(ds_src_dir)

    iterator = TiledDatasetIterator(ds_src, axis="zyx", return_key=True)
    for key, value in iterator:
        print(f"[{key}]")
        print(value)
        print()

    logger.info("dumping destination dataset")
    ds_dst_dir = os.path.join(parent, "data", "demo_3D_2x2x2_CMTKG-V3.zarr")
    ZarrDataset.dump(ds_dst_dir, ds_src)


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    main()
