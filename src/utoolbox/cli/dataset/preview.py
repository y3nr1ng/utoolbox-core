import logging

import click
import numpy as np
import tifffile

import dask.array as da

from utoolbox.io import open_dataset
from utoolbox.io.dataset import (
    MultiChannelDatasetIterator,
    MultiViewDatasetIterator,
    TiledDatasetIterator,
    TimeSeriesDatasetIterator,
)

__all__ = ["preview"]

logger = logging.getLogger("utoolbox.cli.dataset")

VALID_SCREEN_SIZE = {
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
}

VALID_SCREEN_SIZE_CHOICES = list(VALID_SCREEN_SIZE.keys())


def _estaimte_resize_ratio(image, resolution="1440p", portrait=None):
    """
    Estimate the resizing ratio for provided image, in order to generate a sufficient size result for target screen.
    
    Args:
        image (array-like or tuple of int): image to estimate
        resolution (str, optional): screen resolution
        portrait (bool, optional): True if the screen should be in portrait mode, False 
            for landscape, if None, auto infer the orientation
    """
    shape_limit = VALID_SCREEN_SIZE[resolution]
    if portrait is None:
        ny, nx = image.shape
        portrait = ny > nx
    if not portrait:
        # in landscape, screen width > height
        # NOTE since VALID_SCREEN_SIZE store results as (width, height), we need
        # to reverse the shape to C-order if landscape (instead of portrait)
        shape_limit = tuple(shape_limit[::-1])

    shape = image if isinstance(image, tuple) else image.shape
    assert len(shape) == 2, "image is not ready for 2-D screen, programmer error"

    screen_area = shape_limit[0] * shape_limit[1]
    ratio = 1
    while True:
        if all((s // ratio <= s_max) for s_max, s in zip(shape_limit, shape)):
            image_area = (shape[0] // ratio) * (shape[1] // ratio)
            break
        else:
            ratio *= 2
    fill_ratio = image_area / screen_area
    logger.info(f"target downsample ratio {ratio}x, fill ratio {fill_ratio}")

    return ratio


@click.group()
@click.pass_context
def preview(ctx):
    """Generate previews for the dataset."""


@preview.command()
@click.argument("path")
@click.option(
    "-s",
    "--size",
    "screen_size",
    type=click.Choice(VALID_SCREEN_SIZE_CHOICES, case_sensitive=False),
    default="1440p",
    help="what screen size should we fit the result in",
)
@click.pass_context
def mosaic(ctx, path, screen_size):
    """
    Generate mosaic for each layer.
    \f

    Args:
        path (str): path to the dataset    
        size (str, optional): screen size to fit the result in
    """
    show_trace = logger.getEffectiveLevel() <= logging.DEBUG
    ds = open_dataset(path, show_trace=show_trace)

    _, dy, dx = ds.voxel_size

    iz = 0
    for tz, ds_xy in TiledDatasetIterator(ds, axes="z", return_key=True):
        if tz:
            logger.info(f"iterate over z tile, {tz}")

        # populating layers
        layer = []
        for ds_x in TiledDatasetIterator(ds_xy, axes="y", return_key=False):
            row = []
            for uuid in TiledDatasetIterator(ds_x, axes="x", return_key=False):
                row.append(ds[uuid])
            layer.append(row)
        layer = da.block(layer)

        sampler = None
        for mosaic in layer:
            if sampler is None:
                ratio = _estaimte_resize_ratio(mosaic, resolution=screen_size)
                sampler = (slice(None, None, ratio),) * 2
            mosaic = mosaic[sampler]

            print(iz)

            tifffile.imwrite(
                f"mosaic_z{iz:05}.tif",
                mosaic,
                imagej=True,
                resolution=(dx, dy),
                metadata={"unit": "um"},
            )

            iz += 1


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
@click.option("-g", "--gap", type=int, default=1, help="Gap between faces.")
@click.pass_context
def net(ctx, path, gap):
    """Generate net of data blocks."""
    try:
        from utoolbox.util.preview import cuboid_net
    except ImportError:
        logger.error("please install `utoolbox-image` to support surface view")

    show_trace = logger.getEffectiveLevel() <= logging.DEBUG
    ds = open_dataset(path, show_trace=show_trace)

    for time, _ in TimeSeriesDatasetIterator(ds):
        if time is None:
            break
        else:
            raise TypeError(
                "net generation currently does not support time series dataset"
            )

    # calculate scale factor for nets
    scale = _normalized_scale(ds.voxel_size)
    # updated resolution
    res = scale[0] * ds.voxel_size[0]

    desc = ", ".join(f"{k}:{v:.3f}" for k, v in zip("xyz", reversed(scale)))
    logger.debug(f"net scale ({desc}), effective resolution {res:.3f} um")

    # IJ hyperstack order T[Z][C]YXS
    # re-purpose axis meaning:
    #   - Z, slice
    #   - C, channel
    iterator = TiledDatasetIterator(
        ds, axes="zyx", return_key=True, return_format="index"
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
                array = ds[c_ds]
                net = cuboid_net(array, scale, gap=gap)
                nets.append(net)

            # reshape to TZCYXS
            ny, nx = nets[0].shape
            nets = np.stack(nets, axis=0)
            nets.shape = 1, 1, len(nets), ny, nx, 1
            tifffile.imwrite(
                f"{desc}.tif",
                nets,
                imagej=True,
                resolution=(res, res),
                metadata={"unit": "um"},
            )
