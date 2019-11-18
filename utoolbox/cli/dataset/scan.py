import logging
import os

import click
import imageio

from utoolbox.cli.prompt import prompt_options
from utoolbox.data import MicroManagerV1Dataset, MicroManagerV2Dataset, SPIMDataset
from utoolbox.data.dataset.error import DatasetError

from utoolbox.cli.utils import generator

__all__ = ["determine_format"]

logger = logging.getLogger(__name__)


@click.command("scan", short_help="scan and identify a dataset format")
@click.argument("path", type=click.Path(exists=True))
@click.option("-s", "--skip", type=int, default=1, help="number of data to skip")
@generator
def determine_format(path, skip):
    """
    Different dataset are identified and reinterpreted to a common format before going 
    further into the pipeline.
    """
    if os.path.isdir(path):
        datastore = load_from_folder(path)
        for _datastore in datastore:
            for i, key in enumerate(_datastore.keys()):
                if i % skip == 0:
                    logger.info(f".. {key}")
                    yield _datastore[key]
    else:
        data = load_from_file(path)
        if data.ndim == 2:
            yield data
        else:
            # iterate over the slowest axis
            for i in range(data.shape[0]):
                if i % skip == 0:
                    logger.info(f".. {i}")
                    yield data[i, ...]


def load_from_folder(path):
    for klass in (SPIMDataset, MicroManagerV1Dataset, MicroManagerV2Dataset):
        try:
            ds = klass(path)
            break
        except DatasetError:
            logger.debug(f'not "{klass.__name__}"')
    else:
        raise RuntimeError("unable to determine dataset flavor")

    # select channel to work with
    if len(ds.info.channels) > 1:
        channel = prompt_options(
            "Pleae choose a channel to proceed: ", ds.info.channels
        )
    else:
        channel = ds.info.channels[0]

    with ds[channel] as datastore:
        yield datastore


def load_from_file(path):
    yield imageio.volread(path)
