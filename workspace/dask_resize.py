from functools import partial
import glob
import logging
import os

import coloredlogs
from dask import delayed
from dask.cache import Cache
from dask.distributed import Client, fire_and_forget, wait
import imageio
from skimage.transform import downscale_local_mean


def resize_images(dst_dir, src_path):
    data = delayed(imageio.volread, pure=True)(src_path)
    data = delayed(downscale_local_mean, pure=True)(data, (1, 4, 4)).compute()
    fname = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, fname)
    imageio.volwrite(dst_path, data)


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

    src_dir = "S:/ARod/20191212_4F/flybrain_1/camB/128"
    dst_dir = "U:/ARod/20191212_4F/flybrain_1/camB/128"

    file_list = glob.glob(os.path.join(src_dir, "*.tif"))
    logger.info(f"{len(file_list)} file(s) to process")

    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass

    _task = partial(resize_images, dst_dir)
    for file_path in file_list:
        print(file_path)
        resize_images(dst_dir, file_path)
