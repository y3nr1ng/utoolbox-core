# pylint: disable=no-value-for-parameter
"""
Dataset management command-line utility.
"""
import logging

import click
import coloredlogs

logger = logging.getLogger(__name__)

@click.group()
@click.option('-v', '--verbose', count=True)
@click.pass_context
def main(ctx, verbose):
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

@main.command('analyze', short_help="analyze provided data directory")
@click.pass_context
def analyze(ctx, root):
    """TBA"""
    pass

@main.command('preview', short_help="generate preview")
@click.pass_context
def preview(ctx, root):
    pass

if __name__ == '__main__':
    main(obj={})