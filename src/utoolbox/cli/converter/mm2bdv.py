import logging
import os
from shutil import rmtree

import click
import coloredlogs
from prompt_toolkit.shortcuts import button_dialog

from utoolbox.io import open_dataset
from utoolbox.io.dataset import BigDataViewerDataset
from utoolbox.io.dataset.bdv.error import InvalidChunkSizeError

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("src_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "dst_dir",
    help="Output directory, default directory appends `_bdv`.",
)
@click.option(
    "-d",
    "--dry",
    "dry_run",
    is_flag=True,
    default=False,
    help="Dry run, generate XML only.",
)
@click.option(
    "-s",
    "--downsample",
    "downsamples",
    nargs=3,
    type=int,
    multiple=True,
    default=[(1, 1, 1), (4, 4, 2)],
    help='downsample ratio along "X Y Z" axis',
)
@click.option(
    "-c",
    "--chunk",
    "chunk",
    nargs=3,
    type=int,
    multiple=False,
    default=(64, 64, 64),
    help='chunk size along "X Y Z" axis',
)
def main(src_dir, dst_dir, dry_run, downsamples, chunk):
    """
    Convert Micro-Manager dataset to BigDataViewer complient XML/HDF5 format.
    \f

    Args:
        src_path (str): path to the MM dataset
        dst_path (str, optional): where to save the BDV dataset
        dry_run (bool, optinal): save XML only
        downsamples (tuple of int, optional): downsample ratio along (X, Y, Z) axis
        chunk (tuple of int, optional): chunk size
    """
    ds_src = open_dataset(src_dir, show_trace=True)

    if dst_dir is None:
        dst_dir = f"{src_dir}_bdv"

    if os.path.exists(dst_dir):
        dump = button_dialog(
            title="BDV dataset exists",
            text="What should we do?",
            buttons=[("Cancel", False), ("Overwrite", True)],
        ).run()
        if dump:
            # we have to unlink first
            logger.warning("remove previous dataset dump")
            rmtree(dst_dir)
    else:
        dump = True

    if dump:
        # NOTE we should already deal with FileExistError
        os.mkdir(dst_dir)

        # ensure downsamples is wrapped
        if isinstance(downsamples[0], int):
            downsamples = [downsamples]
        # reverse downsampling ratio
        downsamples = [tuple(reversed(s)) for s in downsamples]

        # reverse chunk size
        chunk = tuple(reversed(chunk))

        logger.info("convert to zarr dataset")
        try:
            BigDataViewerDataset.dump(
                dst_dir, ds_src, pyramid=downsamples, chunks=chunk, dry_run=dry_run,
            )
        except InvalidChunkSizeError as err:
            logger.error(str(err))
