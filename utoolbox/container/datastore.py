"""
Create datastore for large collections of data.
"""
from abc import ABCMeta, abstractmethod
import glob
import logging
import os

__all__ = [
    'FileDatastore',
    'ImageDatastore'
]

logger = logging.getLogger(__name__)

class Datastore(object, metaclass=ABCMeta):
    def __init__(self, location, sub_dir=False, pattern='*', extensions=None):
        """
        Parameters
        ----------
        location : str or list of str
            Files or folders to include in the datastore.
        sub_dir : bool, default to False
            Include subfolders within folder.
        extensions : None or list of str
            Extensions of files, select all if 'None'.
        """
        if sub_dir:
            location = os.path.join(location, "**")

        extensions = [pattern if extensions is None else extensions]
        extensions = ["{}.{}".format(pattern, ext) for ext in extensions]

        files = []
        for ext in extensions:
            pathname = os.path.join(location, ext)
            files.extend(glob.glob(pathname, recursive=sub_dir))
        self.files = files

class FileDatastore(Datastore):
    def __init__(self, location, read_func=None, **kwargs):
        super(FileDatastore, self).__init__(location, **kwargs)
        self._read_func = read_func
        self._index = 0
        self._read_size = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.has_data:
            return self.read()
        else:
            raise StopIteration

    @property
    def has_data(self):
        """Determine if data is available to read."""
        return self._index < len(self.files)

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

    @property
    def numpartitions(self):
        """Number of datastore partitions."""
        pass

    def partition(self):
        """Partition a datastore."""
        #TODO mark by labeling
        pass

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
            n_files = len(self.files)
            read_size = self.read_size
            if self._index > n_files:
                read_size -= self._index-n_files
                self._index = n_files
            return [
                self.read_func(self.files[self._index-i-1])
                for i in reversed(range(read_size))
            ]
        else:
            return self.read_func(self.files[self._index-1])

    def read_all(self):
        """Read all data in datastore.

        Note
        ----
        If all the data in the datastore does not fit in memory, then `readall`
        returns an error.
        """
        self._index = len(self.files)
        return [self.read_func(fp) for fp in self.files]

    def reset(self):
        """Reset datastore to initial state."""
        self._index = 0
        self._read_size = 1

class ImageDatastore(FileDatastore):
    supported_extensions = ("tif")

    def __init__(self, location, read_func, extensions=None, **kwargs):
        if extensions is None:
            extensions = ImageDatastore.supported_extensions
        super(ImageDatastore, self).__init__(
            location, read_func=read_func, extensions=extensions, **kwargs
        )
