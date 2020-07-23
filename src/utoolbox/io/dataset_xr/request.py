from pathlib import Path
from typing import Any, Dict, Union

import xarray as xr


class Request:
    """
    Request represents a request for for reading or writing a dataset. This objects 
    wraps information to access a dataset and acts as an interface for the plugins to 
    various resources.

    Args:
        uri (str or Path): the resource to load the dataset from
        mode (str) : indicate how to access the dataset
            - 'r', open file for reading, fail if not exists
            - 'r+', open file for readind and writing, new data appends in the end, 
                which is effectively 'a+'
            - 'w', open file for writing, existing data is truncated, failed if 
                destination does not exist
            - 'w+', similar to 'w', but create the destination if not exist
    """

    def __init__(self, uri: Union[str, Path], mode: str, **kwargs):
        # general
        self._filename = None
        self._extension = None
        self._kwargs = kwargs

        # the actual dataset instance
        self._dataset = None

        # mode string type check
        if not isinstance(mode, str):
            raise ValueError("mode should be a string")
        elif mode[0] not in "rw":
            raise ValueError(
                "mode should start with r (read), w (write), or a (append)"
            )
        elif any(c not in "rw+" for c in mode):
            raise ValueError("mode string contains invalid characters")
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
        """Access mode of this dataset request."""
        return self._mode

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Additional keyword arguments supplied by the user."""
        return self._kwargs

    @property
    def readable(self) -> bool:
        """If True, request a readable dataset."""
        return "r" in self.mode

    @property
    def writable(self) -> bool:
        """If True, request a wriable dataset."""
        return "w" in self.mode

    @property
    def exists(self) -> bool:
        """
        Set to True if the file or directory needs to exist for this request. If False, 
        create new if not exists.
        """
        return self.mode[-1] != "+"

    ##

    def get_dataset(self):
        """
        Get the underlying xarray object that is associated with this dataset. If this 
        is a reading request, the file is in read mode, otherwise, in write mode.

        Args:
            TBD
        """
        # we already have a representation
        if self._dataset is not None:
            return self._dataset

        # TODO

    ##

    def _parse_uri(self, uri):
        self._filename = uri if isinstance(uri, Path) else Path(uri)

        # turn uri to absolute
        self._filename = self._filename.expanduser().resolve()

        # parse extension
        self._extension = "".join(self._filename.suffixes)
