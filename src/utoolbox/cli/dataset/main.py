import logging

import click
import coloredlogs

from .edit import edit
from .export import export
from .info import info
from .preview import preview
from .pyramid import pyramid

__all__ = ["dataset"]

logger = logging.getLogger("utoolbox.cli.dataset")


@click.group()
@click.option("-v", "--verbose", count=True)
@click.pass_context
def dataset(ctx, verbose):
    # we know this is annoying, silence it
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    # convert verbose level
    verbose = 2 if verbose > 2 else verbose
    level = {0: "WARNING", 1: "INFO", 2: "DEBUG"}.get(verbose)
    coloredlogs.install(
        level=level, fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


dataset.add_command(info)
dataset.add_command(preview)
dataset.add_command(pyramid)
dataset.add_command(edit)
dataset.add_command(export)
