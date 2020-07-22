from contextlib import contextmanager

from . import formats
from .dataset import Dataset


@contextmanager
def open_dataset(uri, mode="r", format=None, **kwargs):
    """
    Open a dataset and clean up resource after use automagically.

    Args:
        uri (str or Path): the resource to load the dataset from
        mode (str) : TBD
        **kwargs : TBD
    """
    dataset = Dataset(uri, mode, **kwargs)

    if format is not None:
        # we know the format
        format = formats[format]
        get_reader, get_writer = format.get_reader, format.get_writer
    else:
        # search the format
        get_reader, get_writer = formats.search_read_format, formats.search_write_format

    # TODO use instance attribute to attach reader/writer/indexers
    if "r" in mode:
        dataset.reader = get_reader(dataset, **kwargs)
    if "w" in mode or "a" in mode:
        dataset.writer = get_writer(dataset, **kwargs)

    return dataset
