import logging
import os

import click
import cupy as cp
import imageio

from utoolbox.cli.utils import processor
from utoolbox.util.decorator import run_once

__all__ = ["export"]

logger = logging.getLogger(__name__)


@click.command("export", short_help="export data to file")
@click.option("-f", "--fps", type=float, default=24, help="frame rate")
@click.option("-q", "--quality", type=int, default=8, help="compression quality")
@click.option("-o", "--output", type=click.Path(resolve_path=True))
@click.option("-p", "--prefix", type=str, default="frame_", help="frame prefix")
@click.option("-s", "--suffix", type=str, default="", help="frame suffix")
@click.option("-b", "bigtiff", is_flag=True, default=False, help="BigTIFF mode")
@processor
def export(stream, output, fps, quality, prefix, suffix, bigtiff):
    _, ext = os.path.splitext(output)
    if ext in (".mp4", ".avi"):
        logger.info("export result to a movie container")
        for data in stream:
            writer = get_writer(data, output, fps, quality)
            writer.append_data(cp.asnumpy(data))
            yield data
        writer.close()
    elif ext in (".tif",):
        logger.info("export result to a TIFF stack")
        writer = imageio.get_writer(output, bigtiff=bigtiff)
        for data in stream:
            writer.append_data(cp.asnumpy(data))
            yield data
        writer.close()
    elif ext == "":
        try:
            os.makedirs(output)
        except FileExistsError:
            pass
        logger.info("export result as individual frames")
        i = 0
        for data in stream:
            path = os.path.join(output, f"{prefix}{i}{suffix}.tif")
            imageio.imwrite(path, cp.asnumpy(data))
            i += 1
            yield data
    else:
        raise RuntimeError("unknown file format")


@run_once
def get_writer(data, output, fps, quality):
    """Determine pixel format."""
    px_fmt = "gray" if data.ndim == 2 else "yuv420p"
    return imageio.get_writer(output, fps=fps, quality=quality, pixelformat=px_fmt)

