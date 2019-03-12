# pylint: disable=E1102
from abc import ABC, abstractmethod
from functools import reduce
import logging
import mmap
from operator import mul

import numpy as np

from .error import UnableToConvertError

logger = logging.getLogger(__name__)

__all__ = [
    'Datastore',
    'BufferedDatastore'
]

class Datastore(object):
    """Basic datastore that includes abstract read logic."""
    def __init__(self, read_func=None):
        """
        :param func read_func: function that perform the read operation
        """
        if read_func is None:
            # nop
            self._read_func = lambda x: x
        else:
            self._read_func = read_func
        self._inventory = []
        self._index = 0
        self._read_size = 1
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.has_data:
            return self.read()
        else:
            raise StopIteration

    def __len__(self):
        return len(self._inventory)
        
    @property
    def has_data(self):
        """Determine if data is available to read."""
        return self._index < len(self._inventory)

    @property
    def read_func(self):
        return self._read_func

    @property
    def read_size(self):
        return self._read_size

    @read_size.setter
    def read_size(self, new_read_size):
        if new_read_size < 1:
            raise ValueError("size must be greater than 1")
        else:
            self._read_size = new_read_size

    def preview(self):
        """Subset of data in datastore."""
        curr_index = self._index
        data = self.read()
        self._index = curr_index
        return data

    def read(self):
        """Read data in datastore."""
        self._index += self.read_size
        if self.read_size > 1:
            # avoid overflow during batch read
            n_files = len(self._inventory)
            read_size = self.read_size
            if self._index > n_files:
                read_size -= self._index-n_files
                self._index = n_files
            return [
                self.read_func(self._inventory[self._index-i-1])
                for i in reversed(range(read_size))
            ]
        else:
            return self.read_func(self._inventory[self._index-1])

    def read_all(self):
        """
        Read all data in datastore.

        .. note::
            It attemps to load all the data in memory, be aware of the data 
            size.
        """
        self._index = len(self._inventory)
        return [self.read_func(fp) for fp in self._inventory]

    def reset(self):
        """Reset datastore to initial state."""
        self._index = 0
        self._read_size = 1

class BufferedDatastore(ABC):
    """
    Reading data that requires internal buffer to piece together the fractions before returning it.
    """
    def __init__(self):
        # staging area
        self._mmap, self._buffer = None, None
    
    def __enter__(self):
        shape, dtype = self._buffer_shape()
        nbytes = dtype.itemsize * reduce(mul, shape)
        logger.info(
            "dimension {}, {}, {} bytes".format(shape[::-1], dtype, nbytes)
        )

        self._mmap = mmap.mmap(-1, nbytes)
        self._buffer = np.ndarray(shape, dtype, buffer=self._mmap)

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        del self._buffer
        self._buffer = None
        self._mmap.close()
        logger.debug("buffer destroyed")

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