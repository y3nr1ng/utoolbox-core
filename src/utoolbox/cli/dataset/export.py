import logging
import os

import click

from utoolbox.io import open_dataset
from utoolbox.io.dataset.base import SessionDataset, TiledDataset
from utoolbox.util.logging import change_logging_level

__all__ = ["export"]

logger = logging.getLogger("utoolbox.cli.dataset")


@click.group()
@click.pass_context
def export(ctx):
    """Export info or partial dataset."""


@export.command()
@click.argument("ds_path", metavar="DATASET")
@click.argument("csv_path", metavar="OUTPUT")
@click.option(
    "-p", "--precision", type=int, default=4, help="maximum number of the decimal place"
)
@click.pass_context
def coords(ctx, ds_path, csv_path, precision=4):
    """
    Export filename-coordinate mapping.
    \f

    Args:
        ds_path (str): path to the dataset
        csv_path (str): where to dump the CSV output
        precision (int, optional): maximum number of the decimal place
    """

    show_trace = logger.getEffectiveLevel() <= logging.DEBUG
    ds = open_dataset(ds_path, show_trace=show_trace)

    if not isinstance(ds, TiledDataset):
        raise TypeError("only tiled dataset contains coordinate information")
    if isinstance(ds, SessionDataset):
        raise ValueError("session-based dataset cannot cherry pick internal arrays")

    # reload dataset with alterantive class
    class DumpFilename(type(ds)):
        @property
        def read_func(self):
            def func(uri, shape, dtype):
                return uri

            return func

    logger.debug("reload with DumpFilename")
    with change_logging_level(logging.ERROR):
        ds = DumpFilename.load(ds_path)

    # iterate over uuid and re-interpret the result
    logger.info("mapping UUID to actual filename")
    inventory = ds.inventory.reset_index(name="uuid")
    filenames = [ds[uuid] if uuid else "" for uuid in inventory["uuid"]]
    inventory["filename"] = filenames
    # for multi-file stacks, we explode to expand lists to separate rows
    inventory = inventory.explode("filename")
    # drop uuid column
    inventory.drop("uuid", axis="columns", inplace=True)

    # extract real world coordinate
    coords = ds.tile_coords.reset_index()

    # merge tables
    index_col_names = [name for name in coords.columns if name.startswith("tile_")]
    inventory = inventory.merge(coords, how="left", on=index_col_names)

    # rename columns
    ax = [k.split("_")[1] for k in index_col_names]
    inventory.rename(
        columns={k: f"i{v}" for k, v in zip(index_col_names, ax)}, inplace=True
    )
    inventory.rename(columns={f"{k}_coord": k for k in ax}, inplace=True)

    inventory.to_csv(
        csv_path,
        sep=",",
        index=False,  # ignore row number
        header=True,  # we need column headers
        float_format=f"%.{precision}f",  # 4 digit decimals
    )
