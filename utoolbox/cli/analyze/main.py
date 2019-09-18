import logging

import coloredlogs

logging.getLogger("tifffile").setLevel(logging.ERROR)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.pass_context
def main(ctx, verbose):
    verbosity = ("WARNING", "INFO", "DEBUG")
    if verbose >= len(verbosity):
        verbose = len(verbosity) - 1
    coloredlogs.install(
        level=verbosity[verbosity],
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

@main.command('psf', short_help='analyze PSF info from a beads-coated slide')
@click.argument('path', type=click.Path(exists=True))
@click.pass_context
def analyze(path):
    pass

if __name__ == "__main__":
    main(obj={})
