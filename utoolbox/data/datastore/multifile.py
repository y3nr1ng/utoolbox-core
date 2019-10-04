"""
Datastores that use multiple files to composite a single data entry.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from multiprocessing import cpu_count

import numpy as np

from utoolbox.data.datastore.base import BufferedDatastore
from utoolbox.data.datastore.direct import FolderDatastore

logger = logging.getLogger(__name__)

__all__ = [
    "FolderCollectionDatastore",
    "SparseVolumeDatastore",
    "SparseTiledVolumeDatastore",
]


class FolderCollectionDatastore(FolderDatastore):
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
            fd = FolderDatastore(
                path, sub_dir=False, pattern=file_pattern, extensions=extensions
            )
            # extract the detailed path
            self._uri[name] = list(fd._uri.values())

    @property
    def root(self):
        return self._root


class SparseVolumeDatastore(FolderCollectionDatastore, BufferedDatastore):
    """
    Multiple volumes represented by folders of images. 
    
    Args:
        tile_shape (tuple, optional): tile shape
        tile_order (str, optional): order of the tiles, in 'C' or 'F'
        max_workers (int, optional): maximum number of workers to fetch the data
    """

    def __init__(self, *args, tile_shape=None, tile_order="C", max_workers=0, **kwargs):
        if ("read_func" not in kwargs) or (kwargs["read_func"] is None):
            raise TypeError("read function must be provided to deduce buffer size")
        super().__init__(*args, **kwargs)

        self._tile_shape, self._tile_order = tile_shape, tile_order

        if max_workers < 1:
            max_workers = cpu_count()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.get_event_loop()

    ##

    @property
    def executor(self):
        return self._executor

    @property
    def loop(self):
        return self._loop

    ##

    def _buffer_shape(self):
        # first layer of an arbitrary stack
        paths = next(iter(self._uri.values()))
        image = self._raw_read_func(paths[0])

        nz, (ny, nx) = len(paths), image.shape
        return (nz, ny, nx), image.dtype

    def _deserialize_to_buffer(self, uri_list):
        async def _deserialize_items_to_buffer():
            tasks = self._generate_deserialization_tasks(uri_list)
            await asyncio.wait(tasks)

        self.loop.run_until_complete(_deserialize_items_to_buffer())
        return self._buffer

    def _generate_deserialization_tasks(self, uri_list):
        def _reader(path, z):
            self._buffer[z, ...] = self._raw_read_func(path)

        return [
            self.loop.run_in_executor(self.executor, _reader, path, z)
            for z, path in enumerate(uri_list)
        ]


class SparseTiledVolumeDatastore(SparseVolumeDatastore):
    """
    Similar to spares volumes, but volumes are spatially tiled.

    Args:
        tile_shape (tuple): tile shape
        tile_order (str): order of the tiles, in 'C' or 'F'
        merge (bool): merge the tiles as one when requested
    """

    def __init__(self, *args, merge=True, **kwargs):
        super().__init__(*args, **kwargs)

        self._merge = merge

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
        except TypeError:
            raise TypeError("unable to determine buffer size due to invalid tile shape")
        if self._merge:
            # merge as a big image
            ny, nx = image.shape
            shape = (nty * ny, ntx * nx)
        else:
            # a stack of images
            shape = (ntx * nty,) + image.shape

        return shape, image.dtype

    def _generate_deserialization_tasks(self, uri_list):
        if self._merge:
            # merge as a big image
            it = np.unravel_index(
                list(range(len(uri_list))), self._tile_shape, order=self._tile_order
            )
            ny, nx = self._raw_read_func(uri_list[0]).shape

            def _reader(path, ity, itx):
                logger.debug(f"reading tile ({ity}, {itx})")
                self._buffer[
                    ity * ny : (ity + 1) * ny, itx * nx : (itx + 1) * nx
                ] = self._raw_read_func(path)

            return [
                self.loop.run_in_executor(self.executor, _reader, path, *tile_pos)
                for tile_pos, path in zip(zip(*it), uri_list)
            ]
        else:
            # a stack of images
            return super()._generate_deserialization_tasks(uri_list)
