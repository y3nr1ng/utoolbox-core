import logging

import click
from PyInquirer import prompt

from utoolbox.data import MicroManagerDataset, SPIMDataset
from utoolbox.data.dataset.error import DatasetError

from utoolbox.cli.utils import generator

__all__ = ["determine_format"]

logger = logging.getLogger(__name__)


@click.command("scan", short_help="scan and identify a dataset format")
@click.argument("path", type=click.Path(exists=True))
@click.option('-s', '--skip', type=int, default=1, help='number of data to skip')
@generator
def determine_format(path, skip):
    for klass in (SPIMDataset, MicroManagerDataset):
        try:
            ds = klass(path)
            break
        except DatasetError:
            logger.debug(f'not "{klass.__name__}"')
    else:
        raise RuntimeError("unable to determine dataset flavor")

    # select channel to work with
    if len(ds.info.channels) > 1:
        questions = [
            {
                "type": "list",
                "name": "channel",
                "message": "Please choose a channel to proceed:",
                "choices": ds.info.channels,
            }
        ]
        channel = prompt(questions)['channel']
    else:
        channel = ds.info.channels[0]

    with ds[channel] as source:
        for i, key in enumerate(source.keys()):
            if i % skip == 0:
                logger.info(f".. {key}")
                yield source[key]
