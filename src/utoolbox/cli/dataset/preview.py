import logging

import click
import numpy as np
from tifffile import imwrite

from utoolbox.io import open_dataset
from utoolbox.io.dataset import (
    MultiChannelDatasetIterator,
    MultiViewDatasetIterator,
    TiledDatasetIterator,
    TimeSeriesDatasetIterator,
)

__all__ = ["preview"]

logger = logging.getLogger("utoolbox.cli.dataset")


@click.group()
@click.pass_context
def preview(ctx):
    """Generate previews for the dataset."""
    pass


@preview.command()
@click.argument("path")
@click.pass_context
def mip(ctx, path):
    """Generate maximum intensity projections."""
    raise NotImplementedError


def _normalized_scale(scale):
    """
    Generate the scaling factor for each dimension based on voxel size and array 
    dimension.
    """
    den = min(scale)
    return tuple(s / den for s in scale)


@preview.command()
@click.argument("path")
@click.pass_context
def net(ctx, path):
    """Generate net of data blocks."""
    try:
        from utoolbox.util.preview import cuboid_net
    except ImportError:
        logger.error("please install `utoolbox-image` to support surface view")

    show_trace = logger.getEffectiveLevel() <= logging.DEBUG
    src_ds = open_dataset(path, show_trace=show_trace)

    for time, _ in TimeSeriesDatasetIterator(src_ds):
        if time is None:
            break
        else:
            raise ValueError("surface preview does not support time series dataset")

    # calculate scale factor for nets
    scale = _normalized_scale(src_ds.voxel_size)
    # updated resolution
    res = scale[0] * src_ds.voxel_size[0]

    desc = ", ".join(f"{k}:{v:.3f}" for k, v in zip("xyz", reversed(scale)))
    logger.debug(f"net scale ({desc}), effective resolution {res:.3f} um")

    # IJ hyperstack order T[Z][C]YXS
    # re-purpose axis meaning:
    #   - Z, slice
    #   - C, channel
    iterator = TiledDatasetIterator(
        src_ds, axes="zyx", return_key=True, return_format="index"
    )
    for tile, t_ds in iterator:
        tile_desc = "-".join(
            f"{label}{ax:03d}" for label, ax in zip("xyz", reversed(tile))
        )
        desc = f"tile-{tile_desc}"
        for view, v_ds in MultiViewDatasetIterator(t_ds):
            if view:
                desc = f"view-{view}_{desc}"
            nets = []
            for channel, c_ds in MultiChannelDatasetIterator(v_ds):
                array = src_ds[c_ds]
                net = cuboid_net(array, scale)
                nets.append(net)

            # reshape to TZCYXS
            ny, nx = nets[0].shape
            nets = np.stack(nets, axis=0)
            nets.shape = 1, 1, len(nets), ny, nx, 1
            print(nets.shape)
            imwrite(
                f"{desc}.tif", nets, imagej=True, resolution=(res, res), metadata={},
            )
            # raise RuntimeError("DEBUG, generate 1 tile, 1 view, complete channel")
