from typing import Union
from pathlib import Path


class Request:
    """
    Request represents a unified interface for reading or writing a dataset. This objects wraps information to access a dataset and acts as an interface for the plugins to various resources.

    Args:
        uri (str or Path): the resource to load the dataset from
        mode (str) : TBD
    """

    def __init__(self, uri: Union[str, Path], mode: str, **kwargs):
        pass

    ##

    @property
    def filename(self):
        """
        The uri for which read/write operation was requested. This can be a 
        string-based filename or any other resource identifier.

        Do not rely on the filename to obtain data.
        """
        pass

    @property
    def mode(self):
        """
        The mode of the request. 

        TBD
        """

    ##

    def get_file(self):
        """
        Get a file object for the resource associated with this request. If this is a reading request, the file is in read mode, otherwise, in write mode.
        """
