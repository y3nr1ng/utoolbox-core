import logging
import os

import click
import coloredlogs
from dask.distributed import Client
from prompt_toolkit.shortcuts import button_dialog

from utoolbox.io import open_dataset
from utoolbox.io.dataset import ZarrDataset
from utoolbox.io.dataset.base import TiledDataset

__all__ = ["aszarr"]

logger = logging.getLogger("utoolbox.cli.aszarr")


def _remap_and_flip(ds, remap, flip):
    if not isinstance(ds, TiledDataset):
        # not a tiled dataset
        return ds

    if len(remap) > 1 and remap != "xyz"[: len(remap)]:
        remap = {a: b for a, b in zip("xyz", remap)}
        ds.remap_tiling_axes(remap)
    if flip:
        ds.flip_tiling_axes(list(flip))
    return ds


@click.command()
@click.argument("path", metavar="SOURCE")
@click.option("-v", "--verbose", count=True)
@click.option(
    "-r",
    "--remap",
    type=str,
    default="xyz",
    metavar="AXES",
    help="Reorder axes that does not follow XYZ order.",
)
@click.option(
    "-f",
    "--flip",
    type=str,
    default="",
    metavar="AXES",
    help="Flip axes that are mirrored.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(resolve_path=True),
    default=None,
    help="Destination for the new Zarr dataset.",
)
@click.option(
    "-h",
    "--host",
    "client",
    type=str,
    default=None,
    metavar="HOST",
    help="Cluster scheduler to perform the conversion.",
)
def aszarr(path, verbose, remap, flip, client, output):
    """
    Convert arbitrary dataset into Zarr dataset format.

    If OUTPUT is not specified, it will default to 'SOURCE.zarr'
    \f

    Args:
        path (str): path to the original dataset
        verbose (str, optional): how verbose should the logger behave
        output (str, optional): path to the destination
    """
    # we know this is annoying, silence it
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    # convert verbose level
    level = {0: "WARNING", 1: "INFO", 2: "DEBUG"}.get(verbose, "INFO")
    coloredlogs.install(
        level=level, fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    # ensure we does not have ambiguous input
    src_path = os.path.abspath(path)

    logger.info("loading source dataset")
    show_trace = logger.getEffectiveLevel() <= logging.DEBUG
    ds = open_dataset(src_path, show_trace=show_trace)
    ds = _remap_and_flip(ds, remap, flip)

    # generate the output
    if output is None:
        parent, dname = os.path.split(src_path)
        dst_path = os.path.join(parent, f"{dname}.zarr")
    else:
        dst_path = output
    logger.info(f'converted dataset will save to "{dst_path}"')

    dump, overwrite = True, False
    if os.path.exists(dst_path):
        # output already exists, ask user what's next
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
        desc = "start dumping"
        if client:
            client = Client(client)
            desc += f' (using scheduler "{client.scheduler.address}")'
        logger.info(desc)

        try:
            ZarrDataset.dump(dst_path, ds, overwrite=overwrite, client=client)
        finally:
            if client:
                client.close()

    logger.info("complete zarr dataset conversion")
