import logging
import os
from pprint import pprint

import pandas as pd
from dask.distributed import Client
from prompt_toolkit.shortcuts import button_dialog

from utoolbox.io import open_dataset
from utoolbox.io.dataset import (
    TiledDatasetIterator,
    LatticeScopeTiledDataset,
    ZarrDataset,
)

logger = logging.getLogger("test_zarr")


def main2(ds_src_dir, ds_dst_dir, client=None):
    logger.info("loading source dataset")
    ds_src = open_dataset(ds_src_dir, show_trace=True)

    if isinstance(ds_src, LatticeScopeTiledDataset):
        ds_src.remap_tiling_axes({"x": "z", "y": "x", "z": "y"})
        ds_src.flip_tiling_axes(["x", "y"])

    pprint(ds_src.metadata)

    logger.info("dump dataset info")
    for key, value in TiledDatasetIterator(
        ds_src, return_key=True, return_format="both"
    ):
        print(key)
        print(value)
        print()

    with pd.option_context("display.max_rows", None):
        # print(">> tile_coords")
        # print(ds_src.tile_coords)
        # print()
        print(">> inventory")
        print(ds_src.inventory)
        print()

    dump, overwrite = True, False
    if os.path.exists(ds_dst_dir):
        dump, overwrite = button_dialog(
            title="Zarr dataset exists",
            text="What should we do?",
            buttons=[
                ("Skip", (False, None)),
                ("Update", (True, False)),
                ("Overwrite", (True, True)),
            ],
        ).run()
    else:
        dump, overwrite = True, False

    if dump:
        logger.info("convert to zarr dataset")
        ZarrDataset.dump(ds_dst_dir, ds_src, overwrite=overwrite, client=client)

    logger.info("reload destination dataset")
    ds_dst = ZarrDataset.load(ds_dst_dir)

    print(ds_dst.inventory)

    iterator = TiledDatasetIterator(ds_dst, axes="zyx", return_key=True)
    for key, value in iterator:
        print(f"[{key}]")
        print(value)
        print()


def main(ds_src_dir, ds_dst_dir, client=None):
    ds_dst = ZarrDataset.load(ds_dst_dir)

    print(ds_dst.inventory)

    iterator = TiledDatasetIterator(ds_dst, axes="zyx", return_key=True)
    for key, value in iterator:
        print(f"[{key}]")
        print(value)
        print()


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
        # path = os.path.join(cwd, "../data/cell1a_zp6um_20ms_interval_12s")
        path = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
        ds_src_dir = os.path.abspath(path)
        parent, dname = os.path.split(ds_src_dir)
        ds_dst_dir = os.path.join(parent, f"{dname}.zarr")

    # ensure paths are properly expanded
    ds_src_dir = os.path.abspath(os.path.expanduser(ds_src_dir))
    ds_dst_dir = os.path.abspath(os.path.expanduser(ds_dst_dir))

    # client = Client("10.109.20.6:8786")
    # print(client)
    client = None

    main2(ds_src_dir, ds_dst_dir, client=client)
