from typing import Union
from pathlib import Path
from typing import Dict, Any


class Dataset:
    """
    Dataset represents a unified interface for reading or writing a dataset. This objects wraps information to access a dataset and acts as an interface for the plugins to various resources.

    Args:
        uri (str or Path): the resource to load the dataset from
        mode (str) : TBD
    """

    def __init__(self, uri: Union[str, Path], mode: str, **kwargs):
        # general
        self._filename = None
        self._extension = None
        self._kwargs = kwargs

        # the actual dataset instance
        self._data = None

        if not isinstance(mode, str):
            raise ValueError("mode should be a string")
        self._mode = mode

        self._parse_uri(uri)

    ##

    @property
    def filename(self) -> Path:
        """
        The uri for which read/write operation was requested. This can be a 
        string-based filename or any other resource identifier.

        Do not rely on the filename to obtain data.
        """
        return self._filename

    @property
    def extension(self) -> str:
        """
        Extension of the requested filename.
        
        Note:
            This can be an empty string if the request is not based on a filename.
        """
        return self._extension

    @property
    def mode(self) -> str:
        """
        Access mode of this dataset request.

        TBD
        """
        return self._mode

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Additional keyword arguments supplied by the user."""
        return self._kwargs

    ##

    def get_file(self):
        """
        Get a file object for the resource associated with this request. If this is a reading request, the file is in read mode, otherwise, in write mode.
        """

    ##

    def _parse_uri(self, uri):
        self._filename = uri if isinstance(uri, Path) else Path(uri)

        # turn uri to absolute
        self._filename = self._filename.expanduser().resolve()

        # parse extension
        self._extension = "".join(self._filename.suffixes)
