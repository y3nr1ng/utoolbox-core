import logging

import click
import coloredlogs

from utoolbox.cli.dataset.export import export
from utoolbox.cli.dataset.preview import preview_datastore
from utoolbox.cli.dataset.scan import determine_format
from utoolbox.cli.dataset.transform import rotate


__all__ = ["main"]

logging.getLogger("tifffile").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@click.group(chain=True)
@click.option("-v", "--verbose", count=True)
def main(verbose):
    verbosity = ("WARNING", "INFO", "DEBUG")
    if verbose >= len(verbosity):
        verbose = len(verbosity) - 1
    coloredlogs.install(
        level=verbosity[verbose],
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@main.resultcallback()
def process_commands(processors, verbose):
    stream = ()

    # pipe through all stream processors
    for processor in processors:
        stream = processor(stream)

    # evaluate the stream and throw away the item
    for _ in stream:
        pass


main.add_command(determine_format, "scan")
main.add_command(rotate, "rotate")
main.add_command(preview_datastore, "preview")
main.add_command(export, "export")

if __name__ == "__main__":
    main()
