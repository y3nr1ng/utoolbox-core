import logging

import click
import cupy as cp
from cupyx.scipy.ndimage import rotate as _rotate

from utoolbox.cli.utils import processor

__all__ = ["rotate"]

logger = logging.getLogger(__name__)


@click.command("rotate", short_help="rotate the data source")
@click.option("-d", "--degree", type=float, help="rotate the image in degrees")
@processor
def rotate(stream, degree):
    logger.debug(f'rotate {degree} deg')
    for data in stream:
        data = cp.asarray(data)
        data = _rotate(data, degree, order=1, output=data.dtype)

        yield data
