import glob
import logging
import os

import coloredlogs
from dask import delayed
from dask.cache import Cache
from dask.distributed import as_completed, Client
import imageio
import numpy as np


def load_data(src_path):
    return delayed(imageio.volread)(src_path)


def mip(data, axis=0):
    return delayed(np.amax)(data, axis=axis)


def save_data(dst_path, data):
    data = data.compute().astype(np.float32)
    imageio.imwrite(dst_path, data)
    return dst_path


if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    client = Client("10.109.20.6:8786")
    logger.info(client)

    cache = Cache(2e9)
    cache.register()

    src_dir = "U:/Vins/20191213/10x-fly-brain-64246_naked-brain_zp6um"
    dst_dir = "U:/Vins/20191213/10x-fly-brain-64246_naked-brain_zp6um_mip"

    file_list = glob.glob(os.path.join(src_dir, "**", "*.tif"), recursive=True)
    logger.info(f"{len(file_list)} file(s) to process")

    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass

    futures = []
    for file_path in file_list:
        data = load_data(file_path)
        data = mip(data)

        parent, fname = os.path.split(file_path)
        try:
            internal = os.path.relpath(parent, src_dir)
            parent = os.path.join(dst_dir, internal)
            os.makedirs(parent)
            logger.debug(f'"{parent}" created')
        except FileExistsError:
            pass
        dst_path = os.path.join(parent, fname)

        if os.path.exists(dst_path):
            continue

        future = client.submit(save_data, dst_path, data)

        futures.append(future)

    logger.info(f"awaiting {len(futures)} future(s)")

    for future in as_completed(futures):
        dst_path = future.result()
        fname = os.path.basename(dst_path)
        logger.info(f'"{fname}" saved')

    client.close()
