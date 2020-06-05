import logging
import os
from collections import defaultdict
from itertools import zip_longest

import click
import pandas as pd
from humanfriendly import format_timespan
from humanfriendly.tables import format_pretty_table

from utoolbox.io import open_dataset
from utoolbox.io.dataset import (
    MultiChannelDatasetIterator,
    MultiViewDatasetIterator,
    TiledDatasetIterator,
    TimeSeriesDatasetIterator,
)
from utoolbox.io.dataset.base import DenseDataset, SessionDataset, TILE_INDEX_STR

__all__ = ["info"]

logger = logging.getLogger("utoolbox.cli.dataset")


def _extract_setup_keys(ds):
    """Extract all possible keys."""
    setup = defaultdict(set)
    for timestamp, t_ds in TimeSeriesDatasetIterator(ds):
        setup["timestamp"].add(timestamp)
        for channel, c_ds in MultiChannelDatasetIterator(t_ds):
            setup["channel"].add(channel)
            for view, v_ds in MultiViewDatasetIterator(c_ds):
                setup["view"].add(view)
                for tile, _ in TiledDatasetIterator(v_ds, axes="xyz"):
                    setup["tile"].add(tile)

    return {k: sorted(list(v)) for k, v in setup.items()}


def printi(line, prefix="  ", indent=0, **kwargs):
    """Print with explicit indent option."""
    print(prefix * indent, end="")
    print(line, *kwargs)


@click.command()
@click.argument("path")
@click.option(
    "-a", "--all", "show_all", is_flag=True, default=False, help="List all attributes"
)
@click.pass_context
def info(ctx, path, show_all):
    """
    Dump dataset info.

    This script is not designed with performance in mind, it may take sometime to 
    compile all the info to become human-friendly texts.
    \f

    Args:
        path (str): path to the dataset
        show_all (bool, optional): list all attributes
    """

    show_trace = logger.getEffectiveLevel() <= logging.DEBUG
    ds = open_dataset(path, show_trace=show_trace)

    print()

    # basic type
    path = os.path.abspath(path)
    printi(
        f'Dataset in "{os.path.basename(path)}" is a "{type(ds).__name__}"', indent=1
    )
    if isinstance(ds, SessionDataset):
        printi(f'(Using internal path "{ds.path}")', indent=1)

    print()

    # setup
    setup = _extract_setup_keys(ds)
    n_keys = []
    for k, v in setup.items():
        n_v = len(v)
        if n_v > 1:
            k = k + "s"
        n_keys.append(f"{n_v} {k}")
    desc = " / ".join(n_keys)
    printi(desc, indent=1)

    # setup - statitics
    n_missing = (ds.inventory.values == "").sum()
    if n_missing > 0:
        printi(f"({n_missing} missing data)", indent=1)

    # setup - detail
    if show_all:
        rows = []
        for i, values0 in enumerate(setup.values()):
            values = []
            # reformat
            for value in values0:
                if value is None:
                    value = "-"
                elif isinstance(value, pd.Timedelta):
                    value = format_timespan(value)
                values.append(value)
            rows.append(values)
        rows = zip_longest(*rows, fillvalue="")
        table_str = format_pretty_table(rows, setup.keys())
        # add indent
        table_str = table_str.split("\n")
        for row in table_str:
            printi(row, indent=1)

    print()

    # timeseries
    # TODO time information
    
    # tiles
    if setup["tile"][0] is not None:
        m, M = ds.tile_coords.min().to_dict(), ds.tile_coords.max().to_dict()
        ax_names, extents = [], []
        for ax in TILE_INDEX_STR:
            ax = ax.split("_")[1]
            index = f"{ax}_coord"
            if index in m:
                delta = round(M[index] - m[index], 4)
                extents.append(delta)
                ax_names.append(ax)

        # tiles - index
        printi("Number of tiles", indent=1)
        desc = [f"{k}:{v}" for k, v in zip(ax_names, reversed(ds.tile_shape))]
        desc = ", ".join(desc)
        printi(f"({desc})", indent=2)

        print()

        # tiles - coords
        printi("Tile extents (um)", indent=1)
        desc = [f"{k}:{v}" for k, v in zip(ax_names, extents)]
        desc = ", ".join(desc)
        printi(f"({desc})", indent=2)

        print()

    # voxel size
    if isinstance(ds, DenseDataset):
        printi("Voxel size (um)", indent=1)

        voxel_size = [round(v, 4) for v in reversed(ds.voxel_size)]

        desc = [f"{k}:{v}" for k, v in zip("xyz", voxel_size)]
        desc = ", ".join(desc)
        printi(f"({desc})", indent=2)

    print()
