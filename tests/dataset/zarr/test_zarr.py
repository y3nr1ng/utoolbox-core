import logging
import os
from pprint import pprint

import pandas as pd

from utoolbox.io.dataset import (
    LatticeScopeTiledDataset,
    TiledDatasetIterator,
    ZarrDataset,
    open_dataset,
)

logger = logging.getLogger("test_zarr")


def main(ds_src_dir, ds_dst_dir, client=None):
    logger.info("loading source dataset")
    ds_src = open_dataset(ds_src_dir)

    if isinstance(ds_src, LatticeScopeTiledDataset):
        ds_src.remap_tiling_axes({"x": "z", "y": "x", "z": "y"})
        ds_src.flip_tiling_axes(["x", "y"])

    # iterator = TiledDatasetIterator(ds_src, axis="zyx", return_key=True)
    # for key, value in iterator:
    #    print(f"[{key}]")
    #    if not isinstance(value, list):
    #        value = [value]
    #    for v in value:
    #        print(v)
    #        print(f".. {ds_src[v]}")
    #    print()

    pprint(ds_src.metadata)

    logger.info("dump dataset info")
    for key, value in TiledDatasetIterator(
        ds_src, return_key=True, return_format="both"
    ):
        print(key)
        print(value)
        print()

    with pd.option_context("display.max_rows", None):
        print(">> tile_coords")
        print(ds_src.tile_coords)
        print()
        print(">> inventory")
        print(ds_src.inventory)
        print()

    # raise RuntimeError("DEBUG")

    if not os.path.exists(ds_dst_dir):
        logger.info("convert to zarr dataset")
        ZarrDataset.dump(ds_dst_dir, ds_src, overwrite=True, client=client)

    logger.info("reload destination dataset")
    ds_dst = ZarrDataset.load(ds_dst_dir)


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    if False:
        # Case 1)
        ds_src_dir = "Y:/ARod/4F_/20200506_flybrain_no2_mz19_GFP"
        ds_dst_dir = "Y:/ARod/4F_/20200506_flybrain_no2_mz19_GFP.zarr"
    else:
        # Case 2)
        cwd = os.path.dirname(os.path.abspath(__file__))
        # ds_src_dir = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
        path = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
        ds_src_dir = os.path.abspath(path)
        parent, dname = os.path.split(ds_src_dir)
        ds_dst_dir = os.path.join(parent, f"{dname}.zarr")

    main(ds_src_dir, ds_dst_dir)
