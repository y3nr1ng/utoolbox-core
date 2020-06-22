import logging
import os
from pprint import pprint

import imageio
import pandas as pd
from dask.distributed import Client
from prompt_toolkit.shortcuts import button_dialog

from utoolbox.io import open_dataset
from utoolbox.io.dataset import (
    LatticeScopeTiledDataset,
    MutableZarrDataset,
    TiledDatasetIterator,
    ZarrDataset,
)

logger = logging.getLogger("test_zarr")


def test_dump(ds_src_dir, ds_dst_dir, client=None):
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


def test_load(ds_src_dir, ds_dst_dir, client=None):
    ds_dst = ZarrDataset.load(ds_dst_dir)

    print(ds_dst.inventory)

    iterator = TiledDatasetIterator(ds_dst, axes="zyx", return_key=True)
    for key, value in iterator:
        print(f"[{key}]")
        print(value)
        print()

    print(f'active="{ds_dst.active_label}"')


def test_mutable(ds_src_dir, ds_dst_dir, client=None):
    print(ds_dst_dir)
    if False:
        ds = MutableZarrDataset.load(ds_dst_dir)
    else:
        ds = ZarrDataset.load(ds_dst_dir)
        ds = MutableZarrDataset.from_immutable(ds)

    print(type(ds).__name__)
    print(ds.inventory)

    store_as = "mip_xy"

    iterator = TiledDatasetIterator(ds, axes="zyx", return_key=True)
    for key, uuid in iterator:
        print(f"[{key}]")
        print(uuid.inventory)
        print()

        array = ds[uuid]

        ds.active_label = store_as
        ds[uuid] = array.max(axis=0)

    # reload
    logger.info(f'reload dataset with "{store_as}"')
    ds = ZarrDataset.load(ds_dst_dir, label=store_as)

    ds.remap_tiling_axes({"x": "y", "y": "x"})
    #ds.flip_tiling_axes(["x"])

    iterator = TiledDatasetIterator(ds, axes="xyz", return_key=False)
    for i, uuid in enumerate(iterator):
        print(f"[{key}]")
        print(uuid.inventory)

        data = ds[uuid]

        print(data.shape)
        print()

        filename = f"tile_{i:02d}.tif"
        imageio.imwrite(filename, data)


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
        # path = os.path.join(cwd, "../data/cell1a_zp6um_20ms_interval_12s")
        # path = os.path.join(cwd, "../data/demo_3D_2x2x2_CMTKG-V3")
        path = os.path.join(cwd, "data/ExM_E15_olympus4X_canon300mm_2x3_z20_1")
        ds_src_dir = os.path.abspath(path)
        parent, dname = os.path.split(ds_src_dir)
        ds_dst_dir = os.path.join(parent, f"{dname}.zarr")

    # ensure paths are properly expanded
    ds_src_dir = os.path.abspath(os.path.expanduser(ds_src_dir))
    ds_dst_dir = os.path.abspath(os.path.expanduser(ds_dst_dir))

    # client = Client("10.109.20.6:8786")
    # print(client)
    client = None

    test_mutable(ds_src_dir, ds_dst_dir, client)
