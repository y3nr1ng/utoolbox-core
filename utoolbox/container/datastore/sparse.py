"""
Datastores that represent sparse collections of stacks.
"""
import logging
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
        """
        :param str root: root folder path
        :param read_func: read function for the actual file
        """
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

    def _extract_depth(self, fn, pattern=r'.*_(\d{3,})\.'):
        return int(re.search(pattern, fp).group(1))

    def _find_max_depth(self):
        """Determine depth by one of the stack."""
        src_dir = self.files[0]
        layers = [
            self._extract_depth(fp)
            for fp in filter(lambda x: x.startswith(src_dir), self._raw_files)
        ]
        return max(layers)-min(layers)+1

    def _buffer_shape(self):
        im = self._raw_read_func(self._raw_files[0])
        (ny, nx), nz = im.shape, self.nz

        return (nz, ny, nx), im.dtype

    def _load_to_buffer(self, fn):
        layers = list(filter(lambda x: x.startswith(fn), self._raw_files))
        layers.sort()
        for iz, fp in enumerate(layers):
            self._buffer[iz, ...] = self._raw_read_func(fp)
        return self._buffer

class SparseTilesImageDatastore(SparseStackImageDatastore):
    """Each folder represents a tiled stack."""
    def __init__(self, root, read_func, tile_sz=None, **kwargs):
        """
        :param str root: root folder path
        :param read_func: read function for the actual file
        :param tuple(int,int) tile_sz: dimension of the tiles

        .. note:: Currently, only 2D tiling is supported.
        """
        super(SparseTilesImageDatastore, self).__init__(
            root, read_func, **kwargs
        )
        self._tile_sz = tile_sz if tile_sz else self._find_tile_sz()

    @property
    def tile_sz(self):
        return self._tile_sz

    def _extract_tile_pos(self, fn, pattern=r'.*_(\d{3,})_(\d{3,})'):
        tokens = re.search(pattern, fn)
        return int(tokens.group(2)), int(tokens.group(1))

    def _find_tile_sz(self):
        pos = []
        for fn in self.files:
            try:
                pos.append(self._extract_tile_pos(fn))
            except:
                logger.warning("unknown stack name \"{}\", ignored".format(fn))
                continue
        
        ypos, xpos = list(zip(*pos))
        def find_range(lst):
            return max(lst)-min(lst)+1
        tile_sz = find_range(ypos), find_range(xpos)
        logger.info("tile size {}".format(tile_sz[::-1]))
        
        return tile_sz

    def _buffer_shape(self):
        im = self._raw_read_func(self._raw_files[0])
        ny, nx = im.shape
        nty, ntx = self.tile_sz

        shape = ny*nty, nx*ntx
        
        return (shape, im.dtype)
    
    def _load_to_buffer(self, z, pattern=r'.*_(\d{3,})\.'):
        shape = None
        for dp in self.files:
            ty, tx = self._extract_tile_pos(dp)
            stack = list(filter(lambda x: x.startwith(dp), self._raw_files))
            #TODO build lookup table instead of search all the times
            for fp in stack:
                if self._extract_depth(fp) == z:
                    im = self._raw_read_func(fp)

                    try:
                        ny, nx = shape
                    except TypeError:
                        shape = im.shape
                        ny, nx = shape
                    sel = (slice(ny*ty, ny*(ty+1)), slice(nx*tx, nx*(tx+1)))
                    # TODO fix this vvvvv
                    #self._buffer[*sel] = im

                    break
