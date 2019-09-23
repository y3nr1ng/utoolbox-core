import logging

import click
import coloredlogs

from utoolbox.data.datastore import FolderDatastore
from utoolbox.transform.projections import Orthogonal

from utoolbox.clik.dataset.scan import determine_format
from utoolbox.cli.dataset.preview import preview_dataset

__all__ = ["main"]

logging.getLogger("tifffile").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.pass_context
def main(ctx, verbose):
    verbosity = ("WARNING", "INFO", "DEBUG")
    if verbose >= len(verbosity):
        verbose = len(verbosity) - 1
    coloredlogs.install(
        level=verbosity[verbose],
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


main.add_command(determine_format, "scan")
main.add_command(preview_dataset, "preview")

if __name__ == "__main__":
    main(obj={})
