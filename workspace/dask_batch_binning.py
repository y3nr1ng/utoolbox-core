import glob
import logging
import os

import click
import dask.array as da
import imageio
import numpy as np
from dask import delayed
from dask.distributed import as_completed
import coloredlogs
from utoolbox.util.dask import get_client

logger = logging.getLogger("dask_batch_binning")

# we know this is annoying, silence it
logging.getLogger("tifffile").setLevel(logging.ERROR)

# convert verbose level
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


def main(src_dir, binning=(4, 4, 1), host=None, n_workers=1):
    desc_bin = "-".join(str(b) for b in binning)
    parent, dname = os.path.split(src_dir)
    dst_dir = os.path.join(parent, f"{dname}_bin{desc_bin}")
    logger.info(f'destination: "{dst_dir}"')

    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        logger.warning(f"destination folder already exists")

    file_list = glob.glob(os.path.join(src_dir, "*.tif"))
    logger.info(f"found {len(file_list)} file(s) to process")

    # test read
    array = imageio.volread(file_list[0])
    shape, dtype = array.shape, array.dtype
    del array
    logger.info(f"array info {shape}, {dtype}")

    # build sampler
    sampler = tuple(slice(None, None, b) for b in reversed(binning))

    @delayed
    def volread_np(uri):
        return np.array(imageio.volread(uri))

    def volread_da(uri):
        return da.from_delayed(volread_np(uri), shape, dtype)

    tasks = []
    for src_path in file_list:
        array = volread_da(src_path)
        array = array[sampler]

        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        task = delayed(imageio.volwrite)(dst_path, array)

        tasks.append(task)

    with get_client(address=host, n_workers=n_workers) as client:
        batch_size = len(client.ncores())

        n_failed = 0
        for i in range(0, len(tasks), batch_size):
            futures = [client.compute(task) for task in tasks[i : i + batch_size]]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    logger.exception("something wrong")
                    n_failed += 1
        if n_failed > 0:
            logger.error(f"{n_failed} failed tasks")


@click.command
@click.argument("src_dir", type=str)
@click.option("-b", "--binning", nargs=3, type=int, default=(4, 4, 1))
@click.option("-h", "--host", type=str, default="slurm")
@click.option("-n", "--n_workers", type=int, default=4)
def cli_main(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == "__main__":

    main(
        "C:/Users/Andy/Desktop/utoolbox/utoolbox-core/workspace/data/20200704_kidney_demo-2_CamA",
        (4, 4, 1),
        "localhost:8786",
        4,
    )
