import logging

import click
import cupy as cp
from cupyx.scipy.ndimage import zoom
import numpy as np

from utoolbox.exposure import auto_contrast
from utoolbox.cli.utils import processor
from utoolbox.util.decorator import run_once

__all__ = ["preview_datastore"]

logger = logging.getLogger(__name__)


@click.command("preview", short_help="generate preview")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["zstack", "mip"]),
    default="zstack",
)
@click.option(
    "-s", "--size", type=click.Choice(["4K", "QHD", "FHD", "HD"]), default="FHD"
)
@processor
def preview_datastore(stream, method, size):
    # lookup size
    size = {
        "4K": (2160, 3840),
        "QHD": (1440, 2560),
        "FHD": (1080, 1920),
        "HD": (720, 1280),
    }[size]

    if method == "zstack":
        for data in stream:
            data = cp.asarray(data)

            factor = shape_to_zoom_factor(data.shape, size)
            data = zoom(data, factor, order=1, output=data.dtype)

            data = data.astype(cp.uint16)
            data = auto_contrast(data)
            data = (data - data.min()) / (data.max() - data.min()) * 255
            data = data.astype(cp.uint8)    

            # frame
            yield data

    elif method == "mip":
        result = None
        for data in stream:
            data = cp.asarray(data) 
            try:
                np.maximum(result, data, out=result)
            except TypeError:
                result = data.copy()
        yield result

@run_once
def shape_to_zoom_factor(in_shape, out_shape):
    return min(o / i for i, o in zip(in_shape, out_shape))
