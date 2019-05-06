# pylint: disable=no-value-for-parameter
import logging

import click
import coloredlogs

logger = logging.getLogger(__name__)

@click.command()
@click.option('-v', '--verbose', count=True)
def main(verbose):
    if verbose == 0:
        verbose = 'WARNING'
    elif verbose == 1:
        verbose = 'INFO'
    else:
        verbose = 'DEBUG'
    coloredlogs.install(
        level=verbose,
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )

if __name__ == '__main__':
    main()