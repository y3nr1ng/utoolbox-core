import logging
import os
from pprint import pprint

import pandas as pd
from dask.distributed import Client
from prompt_toolkit.shortcuts import button_dialog

from utoolbox.io.dataset import (
    LatticeScopeTiledDataset,
    TiledDatasetIterator,
    ZarrDataset,
    open_dataset,
)

logger = logging.getLogger("test_zarr")

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

    if True:
        # Case 1)
        ds_src_dir = "X:/charm/20200424_ExM_Thy1_testis_dapi_z2um_1"
        ds_dst_dir = "U:/charm/20200424_ExM_Thy1_testis_dapi_z2um_1.zarr"
    else:
        # Case 2)
        cwd = os.path.dirname(os.path.abspath(__file__))
        # ds_src_dir = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
        path = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
        ds_src_dir = os.path.abspath(path)
        parent, dname = os.path.split(ds_src_dir)
        ds_dst_dir = os.path.join(parent, f"{dname}.zarr")

    # ensure paths are properly expanded
    ds_src_dir = os.path.abspath(os.path.expanduser(ds_src_dir))
    ds_dst_dir = os.path.abspath(os.path.expanduser(ds_dst_dir))

    client = Client("10.109.20.6:8786")
    print(client)

    main(ds_src_dir, ds_dst_dir, client=client)
