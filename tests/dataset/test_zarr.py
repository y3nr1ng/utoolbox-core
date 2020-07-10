import logging
import os
from pprint import pprint

import imageio

from utoolbox.io import open_dataset
from utoolbox.io.dataset import (
    MutableZarrDataset,
    TiledDatasetIterator,
    ZarrDataset,
)
from utoolbox.util.dask import get_client

logger = logging.getLogger("test_zarr")


def test_dump_from_latticescope(ds_src_dir, ds_dst_dir, overwrite=False):
    logger.info("loading source dataset")
    ds_src = open_dataset(ds_src_dir, show_trace=True)

    ds_src.remap_tiling_axes({"x": "z", "y": "x", "z": "y"})
    ds_src.flip_tiling_axes(["x", "y"])

    pprint(ds_src.metadata)

    with get_client(address="localhost:8786", auto_spawn=False):
        logger.info("convert to zarr dataset")
        ZarrDataset.dump(ds_dst_dir, ds_src, overwrite=overwrite)


def test_load(ds_src_dir, ds_dst_dir, client=None):
    ds_dst = ZarrDataset.load(ds_dst_dir)

    print(ds_dst.labels)


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
        if False:
            del ds[uuid]
        else:
            ds[uuid] = array.max(axis=0)

    return
    # reload
    logger.info(f'reload dataset with "{store_as}"')
    ds = ZarrDataset.load(ds_dst_dir, label=store_as)

    ds.flip_tiling_axes(["y"])

    iterator = TiledDatasetIterator(ds, axes="zyx", return_key=False)
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

    ds_src_dir = "c:/users/andy/desktop/utoolbox/utoolbox-core/workspace/data/20200704_kidney_demo-2_CamA"
    ds_dst_dir = f"{ds_src_dir}.zarr"

    test_dump_from_latticescope(ds_src_dir, ds_dst_dir)
