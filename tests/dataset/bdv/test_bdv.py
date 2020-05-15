import logging
import os
from pprint import pprint
from shutil import rmtree

import pandas as pd
from dask.distributed import Client
from prompt_toolkit.shortcuts import button_dialog

from utoolbox.io import open_dataset
from utoolbox.io.dataset import BigDataViewerDataset, TiledDatasetIterator

logger = logging.getLogger("test_zarr")


def main(ds_src_dir, ds_dst_dir, client=None):
    logger.info("loading source dataset")
    ds_src = open_dataset(ds_src_dir)

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

    if os.path.exists(ds_dst_dir):
        dump = button_dialog(
            title="BDV dataset exists",
            text="What should we do?",
            buttons=[("Skip", False), ("Overwrite", True),],
        ).run()
        if dump:
            # we have to unlink first
            logger.warning("remove previous dataset dump")
            rmtree(ds_dst_dir)
    else:
        dump = True

    if dump:
        logger.info("convert to zarr dataset")
        BigDataViewerDataset.dump(
            ds_dst_dir, ds_src, pyramid=[(1, 1, 1), (2, 4, 4)], chunks=(16, 128, 128)
        )


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    if False:
        # Case 1)
        ds_src_dir = "X:/charm/20200424_ExM_Thy1_testis_dapi_z2um_1"
        ds_dst_dir = "U:/charm/20200424_ExM_Thy1_testis_dapi_z2um_1.zarr"
    else:
        # Case 2)
        cwd = os.path.dirname(os.path.abspath(__file__))
        # ds_src_dir = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
        path = os.path.join(cwd, "../data/ExM_E15_olympus4X_canon300mm_2x3_1")
        ds_src_dir = os.path.abspath(path)
        parent, dname = os.path.split(ds_src_dir)
        ds_dst_dir = os.path.join(parent, f"{dname}_bdv")

    # ensure paths are properly expanded
    ds_src_dir = os.path.abspath(os.path.expanduser(ds_src_dir))
    ds_dst_dir = os.path.abspath(os.path.expanduser(ds_dst_dir))

    # client = Client("10.109.20.6:8786")
    # print(client)
    client = None

    main(ds_src_dir, ds_dst_dir, client=client)
