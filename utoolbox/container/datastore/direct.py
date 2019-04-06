"""
Datastores that can direct access the files.
"""
import glob
import logging
import os
import re

from .base import Datastore

logger = logging.getLogger(__name__)

__all__ = [
    'FileDatastore',
    'ImageDatastore'
]

class FileDatastore(Datastore):
    def __init__(self, root, read_func=None, sub_dir=False, pattern='*', 
                 extensions=None):
        """
        :param str root: files or folders to include in the datastore
        :param func read_func: function that perform the read operation
        :param bool sub_dir: scan nested folders
        :param str pattern: filename pattern
        :param str extensions: file extensions to include
        """
        super().__init__(read_func)

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
        FileDatastore._sort_numerically(files)
        self._inventory = files

    @property
    def files(self):
        return self._inventory

    @staticmethod
    def _sort_numerically(files):
        # extract valid numbers from filenames
        keys = [list(map(int, re.findall(r'[0-9]+', fn))) for fn in files]
        # identify constant variables
        flags = [
            all(elem == elems[0] for elem in elems) 
            for elems in zip(*keys)
        ]
        # keep varying keys
        keys[:] = [[k for k, f in zip(key, flags) if not f] for key in keys]
        
        # sort the file list based on extracted keys
        files[:] = [f for _, f in sorted(zip(keys, files))]

class ImageDatastore(FileDatastore):
    supported_extensions = ['tif']

    def __init__(self, root, read_func, extensions=None, **kwargs):
        if extensions is None:
            extensions = ImageDatastore.supported_extensions
        super().__init__(
            root, read_func=read_func, extensions=extensions, **kwargs
        )
            