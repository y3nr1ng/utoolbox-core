from contextlib import contextmanager, nullcontext

from . import formats
from .request import Request
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
    request = Request(uri, mode, **kwargs)

    # determine access format
    if format is not None:
        # we know the format
        format = formats[format]
        r_format, w_format = format, format
    else:
        r_format = formats.search_read_format(request) if request.readable else None
        w_format = formats.search_write_format(request) if request.writable else None

    # get reader/writer
    try:
        reader = r_format.get_reader(request, **kwargs)
    except AttributeError:
        reader = nullcontext()
    try:
        writer = w_format.get_writer(request, **kwargs)
    except AttributeError:
        writer = nullcontext()

    # create dataset instance
    with reader, writer:
        yield Dataset(reader, writer).get_xarray()
