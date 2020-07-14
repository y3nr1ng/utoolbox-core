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
import napari

logger = logging.getLogger("test_napari")


def test_napari(ds_src_dir, ds_dst_dir):
    # store_as = "mip_xy"
    store_as = "raw"

    # reload
    logger.info(f'load dataset with "{store_as}"')
    ds = ZarrDataset.load(ds_dst_dir, label=store_as)

    iterator = TiledDatasetIterator(ds, axes="zyx", return_key=False)
    for i, uuid in enumerate(iterator):
        data = ds[uuid]

        with napari.gui_qt():
            viewer = napari.view_image(data, scale=ds.voxel_size)

        raise RuntimeError("DEBUG")


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    ds_src_dir = "c:/users/andy/desktop/utoolbox/utoolbox-core/workspace/data/20200704_kidney_demo-2_CamA"
    ds_dst_dir = f"{ds_src_dir}.zarr"

    with get_client(address="localhost:8786"):
        test_napari(ds_src_dir, ds_dst_dir)
