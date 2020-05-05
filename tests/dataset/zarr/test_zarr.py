import logging
import os
from pprint import pprint

import pandas as pd

from utoolbox.io.dataset import TiledDatasetIterator, ZarrDataset, open_dataset

logger = logging.getLogger("test_zarr")


def main(ds_src_dir, ds_dst_dir, client=None):
    logger.info("loading source dataset")
    ds_src = open_dataset(ds_src_dir)

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
    with pd.option_context("display.max_rows", None):
        print(ds_src.inventory)

    if not os.path.exists(ds_dst_dir):
        logger.info("dumping destination dataset")
        ZarrDataset.dump(ds_dst_dir, ds_src, overwrite=True, client=client)

    logger.info("reload destination dataset")
    ds_dst = ZarrDataset.load(ds_dst_dir)


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    # Case 1)
    # ds_src_dir = "X:/charm/20200430_ExM_7d_kb-2_nkcc2_488_5um_p11x16_2x2bin_1"
    # ds_dst_dir = "U:/charm/20200430_ExM_7d_kb-2_nkcc2_488_5um_p11x16_2x2bin_1.zarr"

    # Case 2)
    cwd = os.path.dirname(os.path.abspath(__file__))
    # ds_src_dir = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
    path = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
    ds_src_dir = os.path.abspath(path)
    parent, dname = os.path.split(ds_src_dir)
    ds_dst_dir = os.path.join(parent, f"{dname}.zarr")

    main(ds_src_dir, ds_dst_dir)
