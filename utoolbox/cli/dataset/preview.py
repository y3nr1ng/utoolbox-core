import logging

import click
import imageio
import numpy as np
from scipy.ndimage import zoom

from utoolbox.cli.utils import processor

__all__ = ["preview_datastore"]

logger = logging.getLogger(__name__)


@click.command("preview", short_help="generate preview")
@click.option(
    "-s", "--size", type=click.Choice(["4K", "QHD", "FHD", "HD"]), default="4K"
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["zstack", "rotate", "montage"]),
    default="zstack",
)
@click.option("--fps", type=float, default=24)
@click.option("-o", "--output", type=click.Path())
@processor
def preview_datastore(datastore, size, method, fps, output):
    # lookup size
    size = {
        "4K": (2160, 3840),
        "QHD": (1440, 2560),
        "FHD": (1080, 1920),
        "HD": (720, 1280),
    }[size]

    if method == "zstack":
        writer = imageio.get_writer(output, fps=fps)
        for data in datastore:
            m, M = data.min(), data.max()
            data = (data - m) / (M - m)
            data = (data * 255).astype(np.uint8)

            factor = shape_to_zoom_factor(data.shape, size)
            data = zoom(data, factor)

            imageio.imwrite('test.tif', data)

            writer.append_data(data)

            # frame
            yield data
        writer.close()
    elif method == "rotate":
        raise NotImplementedError()
    elif method == "montage":
        raise NotImplementedError()


def shape_to_zoom_factor(in_shape, out_shape):
    return min(o / i for i, o in zip(in_shape, out_shape))
