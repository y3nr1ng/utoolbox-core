from contextlib import contextmanager

from .dataset import Dataset
from ... import formats


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

    # get format
    if format is not None:
        format = formats[format]
    else:
        format = format.search_
