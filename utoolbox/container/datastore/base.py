# pylint: disable=E1102
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping
from functools import reduce
import logging
import mmap
from operator import mul
import sys

import numpy as np

from .error import (
    ImmutableUriListError,
    ReadOnlyDataError
)

logger = logging.getLogger(__name__)

__all__ = [
    'Datastore',
    'BufferedDatastore'
]

class Datastore(MutableMapping):
    """Basic datastore that includes abstract read logic."""
    def __init__(self, read_func=None, write_func=None, immutable=False):
        """
        :param func read_func: function that perform the read operation
        """
        self._uri = OrderedDict()

        if read_func is None:
            # nop
            _read_func = lambda x: x
        else:
            def wrapped_read_func(key):
                try:
                    return read_func(self._uri[key])
                except KeyError:
                    raise FileNotFoundError("unknown key \"{}\"".format(key))
            _read_func = wrapped_read_func
        if write_func is None:
            def raise_readonly_error(key, value):
                raise ReadOnlyDataError("current dataset is read-only")
            _write_func = raise_readonly_error
        else:
            def wrapped_write_func(key, value):
                try:
                    uri = self._uri[key]
                except KeyError:
                    if immutable:
                        raise ImmutableUriListError("datastore is immutable")
                    else:
                        uri = self._key_to_uri(key)
                        self._uri[key] = uri
                write_func(uri, value)
            _write_func = wrapped_write_func
        self._read_func, self._write_func = _read_func, _write_func

    def __delitem__(self, key):
        raise ImmutableInventoryError("cannot delete entries in a datastore")

    def __getitem__(self, key):
        print(key)
        return self.read_func(key)
    
    def __iter__(self):
        return iter(self._uri)

    def __len__(self):
        return len(self._uri)

    def __setitem__(self, key, value):
        self.write_func(key, value)

    @property
    def read_func(self):
        return self._read_func

    @property
    def write_func(self):
        return self._write_func
    
    def _key_to_uri(self, key):
        raise ImmutableUriListError("key transform function not defined")

class BufferedDatastore(Datastore):
    """
    Reading data that requires internal buffer to piece together the fractions before returning it.
    """
    def __init__(self, *args, **kwargs):
        # staging area
        self._mmap, self._buffer = None, None

        super().__init__(*args, **kwargs)
    
    def __enter__(self):
        self._allocate_buffer()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._free_buffer()
        logger.debug("internal buffer destroyed")

    @abstractmethod
    def _buffer_shape(self):
        """
        Determine shape and type of the internal buffer.
        
        :return: a tuple, (shape, dtype)
        """
        raise NotImplementedError
    
    @abstractmethod
    def _load_to_buffer(self, x):
        """
        Load data definition x into the internal buffer.
        
        :param x: any definition that can be successfully interpreted internally
        """
        raise NotImplementedError

    def _allocate_buffer(self):
        shape, dtype = self._buffer_shape()
        nbytes = dtype.itemsize * reduce(mul, shape)
        logger.info(
            "dimension {}, {}, {} bytes".format(shape[::-1], dtype, nbytes)
        )

        self._mmap = mmap.mmap(-1, nbytes)
        self._buffer = np.ndarray(shape, dtype, buffer=self._mmap)

    def _free_buffer(self):
        if sys.getrefcount(self._buffer) > 2:
            # getrefcount + self._buffer -> 2 references
            logger.warning("buffer is referenced externally")
        self._buffer = None
        self._mmap.close()
