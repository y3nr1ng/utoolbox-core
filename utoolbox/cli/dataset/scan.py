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
@generator
def determine_format(path):
    for klass in (SPIMDataset, MicroManagerDataset):
        try:
            ds = klass(path)
            break
        except DatasetError:
            logger.debug(f'not "{klass.__name__}"')
    else:
        raise RuntimeError("unable to determine dataset flavor")

    if len(ds.info.channels) > 1:
        questions = [
            {
                "type": "list",
                "name": "channel",
                "message": "Please choose a channel to proceed:",
                "choices": ds.info.channels,
            }
        ]
        answers = prompt(questions)

        with ds[answers["channel"]] as source:
            i = 0
            for key, value in source.items():
                logger.info(f".. {key}")

                # DEBUG
                if i < 5:
                    i += 1
                else:
                    break

                yield value
