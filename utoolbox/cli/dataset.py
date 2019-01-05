"""
Dataset management command-line utility.
"""
import logging

import click
import coloredlogs

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    pass
    