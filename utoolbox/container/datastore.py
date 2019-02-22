"""
Create datastore for large collections of data.
"""
# pylint: disable=E1102

from abc import ABCMeta, abstractmethod
import gc
import glob
import json
import logging
from lzma import FORMAT_XZ, LZMADecompressor
import mmap
import os
import re
import tarfile
import tempfile

import numpy as np
import xxhash

__all__ = [
    'Datastore',
    'FileDatastore',
    'ImageDatastore',
    'TarDatastore'
]

logger = logging.getLogger(__name__)

class Datastore(object, metaclass=ABCMeta):
    def __init__(self, read_func=None):
        """
        Parameters
        ----------
        read_func : func
            Function to perform the read operation.
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
        """Read all data in datastore.

        Note
        ----
        If all the data in the datastore does not fit in memory, then `readall`
        returns an error.
        """
        self._index = len(self._inventory)
        return [self.read_func(fp) for fp in self._inventory]

    def reset(self):
        """Reset datastore to initial state."""
        self._index = 0
        self._read_size = 1

class FileDatastore(Datastore):
    def __init__(self, root, read_func=None, sub_dir=False, pattern='*', 
                 extensions=None):
        """
        Parameters
        ----------
        root : str or list of str
            Files or folders to include in the datastore.
        read_func : func
            Function to perform the read operation.
        sub_dir : bool, default to False
            Include subfolders within folder.
        pattern : str
            Patterns in the filename, default to '*'.
        extensions : None or list of str
            Extensions of files, select all if 'None'.
        """
        super(FileDatastore, self).__init__(read_func)

        if sub_dir:
            root = os.path.join(root, "**")
        logger.debug("search under \"{}\"".format(root))

        if extensions is None:
            extensions = [pattern]
        else:
            extensions = ["{}.{}".format(pattern, ext) for ext in extensions]
        logger.debug("{} search patterns".format(len(extensions)))

        files = []
        for ext in extensions:
            path = os.path.join(root, ext)
            files.extend(glob.glob(path, recursive=sub_dir))
        self._inventory = files

    @property
    def files(self):
        return self._inventory
    
class ImageDatastore(FileDatastore):
    supported_extensions = ['tif']

    def __init__(self, root, read_func, extensions=None, **kwargs):
        if extensions is None:
            extensions = ImageDatastore.supported_extensions
        super(ImageDatastore, self).__init__(
            root, read_func=read_func, extensions=extensions, **kwargs
        )

class SparseStackImageDatastore(ImageDatastore):
    """Each folder represents a stack."""
    stack_pattern = r'.*-Pos_(\d{3,})_(\d{3,})$'

    def __init__(self, root, read_func, **kwargs):
        stacks = next(os.walk(root))[1]
        stacks.sort()
        logger.debug("found {} stacks".format(len(stacks)))
        # add suffix
        stacks[:] = [os.path.join(root, fp) for fp in stacks]

        # staging area
        self._mmap, self._buffer = None, None
        self._raw_read_func = read_func

        kwargs['sub_dir'] = True
        super(SparseStackImageDatastore, self).__init__(
            root, read_func=self._load_to_buffer, **kwargs
        )
        self._root = root
        
        # overwrite the original file list
        self._inventory, self._raw_files = stacks, self._inventory

        # update depth
        self._nz = self._find_max_depth()

    def __enter__(self):
        self._generate_buffer()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        del self._buffer
        self._buffer = None
        self._mmap.close()
        logger.debug("buffer destroyed")

    @property
    def nz(self):
        return self._nz

    @property
    def root(self):
        return self._root

    def _find_max_depth(self, pattern=r'.*_(\d{3,})\.'):
        """Determine depth by one of the stack."""
        src_dir = self.files[0]
        layers = [
            int(re.search(pattern, fp).group(1))
            for fp in filter(lambda x: x.startswith(src_dir), self._raw_files)
        ]
        return max(layers)-min(layers)+1

    def _generate_buffer(self):
        im = self._raw_read_func(self._raw_files[0])
        (ny, nx), nz = im.shape, self.nz

        nbytes = (nx*ny*nz) * im.dtype.itemsize
        logger.info(
            "dimension {}, {}, {} bytes".format((nx, ny, nz), im.dtype, nbytes)
        )

        self._mmap = mmap.mmap(-1, nbytes)
        self._buffer = np.ndarray((nz, ny, nx), im.dtype, buffer=self._mmap)

    def _load_to_buffer(self, fn):
        layers = list(filter(lambda x: x.startswith(fn), self._raw_files))
        layers.sort()
        for iz, fp in enumerate(layers):
            self._buffer[iz, ...] = self._raw_read_func(fp)
        return self._buffer

class SparseTilesImageDatastore(ImageDatastore):
    """Each folder represents a tile stack."""
    pass

class TarDatastore(Datastore):
    def __init__(self, root, read_func, memlimit=None):
        if not root.lower().endswith('.tar'):
            raise ValueError("not a TAR file")   
        self._root = root

        tar = tarfile.open(self.root, 'r:')
        self._tar = tar
        self._inventory = tar.getmembers()

        # TODO do we need metadata?
        try:
            tarinfo = tar.getmember('metadata')
            metadata = tar.extractfile(tarinfo).read()
            self._metadata = json.load(metadata)
        except KeyError:
            raise InvalidMetadataError()

        self._decompressor = LZMADecompressor(
            format=FORMAT_XZ, memlimit=memlimit
        )
        def _wrapped_read_func(tarinfo):
            """Read from compressed data block."""
            data = self._tar.extractfile(tarinfo).read()
            # TODO decompression
            #data = self._decompressor.decompress(data)

            digest = xxhash.xxh64_hexdigest(data)
            if tarinfo.name != digest:
                raise HashMismatchError("hash mismatch after decompression")

            return read_func(data)
        super(TarDatastore, self).__init__(_wrapped_read_func)

        # non-wrapped read_func
        self._base_read_func = read_func
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._tar.close()

    @property
    def root(self):
        return self._root

class DatastoreError(Exception):
    """Base class for datastore exceptions."""

class InvalidMetadataError(DatastoreError):
    """Invalid metadata in tar datastores."""

class HashMismatchError(DatastoreError):
    """Digest mismatch after file decompression."""