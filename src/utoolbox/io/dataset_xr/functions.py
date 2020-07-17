from contextlib import contextmanager


@contextmanager
def open_dataset(uri, mode, **kwargs):
    """
    Open a dataset and clean up resource after use automagically.

    Args:
        uri (str or Path): the resource to load the dataset from
        mode (str) : TBD
        **kwargs : TBD
    """
