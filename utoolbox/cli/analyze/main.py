import logging

import click
import coloredlogs

from utoolbox.cli.analyze.psf import analyze_psf

__all__ = ["main"]


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


main.add_command(analyze_psf, "psf")


if __name__ == "__main__":
    main(obj={})
