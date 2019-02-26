import itertools
import logging
import mmap
import os
import re

import numpy as np

from .base import BufferedDatastore
from .direct import ImageDatastore

logger = logging.getLogger(__name__)

__all__ = [
    'SparseStackImageDatastore',
    'SparseTilesImageDatastore'
]

class SparseStackImageDatastore(ImageDatastore, BufferedDatastore):
    """Each folder represents a stack."""
    def __init__(self, root, read_func, **kwargs):
        stacks = next(os.walk(root))[1]
        stacks.sort()
        logger.debug("found {} stacks".format(len(stacks)))
        # add suffix
        stacks[:] = [os.path.join(root, fp) for fp in stacks]

        self._raw_read_func = read_func

        kwargs['sub_dir'] = True
        ImageDatastore.__init__(
            self,
            root, read_func=self._load_to_buffer, **kwargs
        )
        BufferedDatastore.__init__(self)
        self._root = root
        
        # overwrite the original file list
        self._inventory, self._raw_files = stacks, self._inventory

        # update depth
        self._nz = self._find_max_depth()

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
        self._mmap = mmap.mmap(-1, nbytes)
        self._buffer = np.ndarray((nz, ny, nx), im.dtype, buffer=self._mmap)

    def _load_to_buffer(self, fn):
        layers = list(filter(lambda x: x.startswith(fn), self._raw_files))
        layers.sort()
        for iz, fp in enumerate(layers):
            self._buffer[iz, ...] = self._raw_read_func(fp)
        return self._buffer

class SparseTilesImageDatastore(SparseStackImageDatastore):
    """Each folder represents a tile stack."""
    def __init__(self, root, read_func, tile_sz=None, **kwargs):
        super(SparseTilesImageDatastore, self).__init__(
            root, read_func, **kwargs
        )
        self._tile_sz = tile_sz if tile_sz else self._find_tile_sz()

    @property
    def tile_sz(self):
        return self._tile_sz

    def _find_tile_sz(self, pattern=r'Pos_(\d{3,})_(\d{3,})'):
        pos = []
        for fn in self.files:
            try:
                tokens = re.search(pattern, fn)
            except:
                logger.warning("unknown stack name \"{}\", ignored".format(fn))
                continue
            pos.append((int(tokens.group(2)), int(tokens.group(1))))
        
        ypos, xpos = list(zip(*pos))
        def find_range(lst):
            return max(lst)-min(lst)+1
        tile_sz = find_range(ypos), find_range(xpos)
        logger.info("tile size {}".format(tile_sz[::-1]))
        
        return tile_sz

    def _generate_buffer(self):
        im = self._raw_read_func(self._raw_files[0])
        ny, nx = im.shape
        nty, ntx = self.tile_sz

        shape = ny*nty, nx*ntx
        nbytes = shape[0] * shape[1] * im.dtype.itemsize
        self._mmap = mmap.mmap(-1, nbytes)
        self._buffer = np.ndarray(shape, im.dtype, buffer=self._mmap)
    
    def _load_to_buffer(self, z):
        #TODO finish this
        
        for i, j in itertools.product(range(0, 5), (6, 9)):
            pass

        layers = list(filter(lambda x: x.startswith(fn), self._raw_files))
        layers.sort()
        for iz, fp in enumerate(layers):
            self._buffer[iz, ...] = self._raw_read_func(fp)
        return self._buffer
