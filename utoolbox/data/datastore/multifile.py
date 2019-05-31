"""
Datastores that use multiple files to composite a single data entry.
"""
import logging

import numpy as np

from .direct import FileDatastore
from .base import BufferedDatastore

logger = logging.getLogger(__name__)

__all__ = ["FolderCollectionDatastore", "VolumeTilesDatastore"]


class FolderCollectionDatastore(FileDatastore):
    """Each folder represents a stack."""

    def __init__(
        self, root, folder_pattern="*", file_pattern="*", extensions=None, **kwargs
    ):
        """
        :param str root: root folder path
        """
        # override, sub-dir scan is postponed
        kwargs.update({"sub_dir": False, "pattern": folder_pattern, "extensions": None})
        super().__init__(root, **kwargs)

        # expand the file list
        for name, path in self._uri.items():
            # treat each folder as a file datastore
            fd = FileDatastore(
                path, sub_dir=False, pattern=file_pattern, extensions=extensions
            )
            # extract the detailed path
            self._uri[name] = list(fd._uri.values())

    @property
    def root(self):
        return self._root


class VolumeTilesDatastore(FolderCollectionDatastore, BufferedDatastore):
    def __init__(self, *args, tile_shape=None, merge=True, **kwargs):
        """
        :param tile_shape: tile shape in 2D
        :param bool merge: merge the tiles as one
        """
        if ("read_func" not in kwargs) or (kwargs["read_func"] is None):
            raise TypeError("read function must be provided to deduce buffer size")
        super().__init__(*args, **kwargs)

        self._tile_shape, self._merge = tile_shape, merge

        # an arbitrary stack to prime the new URI list
        new_uri = {z: [] for z in range(len(next(iter(self._uri.values()))))}
        # iterate over items to redistribute the paths
        for paths in self._uri.values():
            for z, path in enumerate(paths):
                # assuming the original path list is sorted
                new_uri[z].append(path)
        self._uri = new_uri

    def _buffer_shape(self):
        # first layer of an arbitrary stack
        paths = next(iter(self._uri.values()))
        image = self._raw_read_func(paths[0])

        try:
            nty, ntx = self._tile_shape
        except:
            raise ValueError(
                "unable to determine buffer size due to invalid tile shape"
            )
        if self._merge:
            ny, nx = image.shape
            shape = (nty * ny, ntx * nx)
        else:
            shape = (ntx * nty,) + image.shape

        return shape, image.dtype

    def _deserialize_to_buffer(self, uri_list):
        if self._merge:
            it = np.unravel_index(list(range(len(uri_list))), self._tile_shape)
            ny, nx = self._raw_read_func(uri_list[0]).shape
            for (ity, itx), path in zip(it, uri_list):
                im = self._raw_read_func(path)
                self._buffer[ity * ny : (ity + 1) * ny, itx * nx : (itx + 1) * nx] = im
        else:
            np.concatenate(
                [self._raw_read_func(path) for path in uri_list], out=self._buffer
            )


'''
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
'''
