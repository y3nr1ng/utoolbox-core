import logging

import click
import cupy as cp
import imageio

from utoolbox.cli.utils import processor
from utoolbox.utils.decorator import run_once

__all__ = ["export"]

logger = logging.getLogger(__name__)


@click.command("export", short_help="export data to file")
@click.option("-f", "--fps", type=float, default=24, help="frame rate")
@click.option("-q", "--quality", type=int, default=8, help="compression quality")
@click.option("-o", "--output", type=click.Path(writable=True, resolve_path=True))
@processor
def export(stream, fps, quality, output):
    for data in stream:
        writer = get_writer(data, output, fps, quality)      
        writer.append_data(cp.asnumpy(data))
        yield data
    writer.close()


@run_once
def get_writer(data, output, fps, quality):
    """Determine pixel format."""
    px_fmt = "gray" if data.ndim == 2 else "yuv420p"
    return imageio.get_writer(output, fps=fps, quality=quality, pixelformat=px_fmt)
