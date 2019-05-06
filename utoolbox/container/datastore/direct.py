"""
Datastores that can direct access the files.
"""
import glob
import logging
import os
import re

from .base import Datastore
from .error import InvalidDatastoreRootError

logger = logging.getLogger(__name__)

__all__ = [
    'FileDatastore',
    'ImageDatastore'
]

class FileDatastore(Datastore):
    def __init__(self, root, sub_dir=False, pattern='*', extensions=None,
                 create_new=True, **kwargs):
        """
        :param str root: files or folders to include in the datastore
        :param bool sub_dir: scan nested folders
        :param str pattern: filename pattern
        :param str extensions: file extensions to include
        :param bool create_new: create datastore root if not exists
        """
        super().__init__(**kwargs)

        if not os.path.exists(root):
            if create_new:
                os.mkdir(root)
            else:
                raise InvalidDatastoreRootError(
                    "unable to find \"{}\"".format(root)
                )
            # short circuit
            return
        self._root = root

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

        # simple 1-1 mapping
        for path in files:
            self._uri[os.path.basename(path)] = path

    @property
    def root(self):
        return self._root

    @staticmethod
    def _sort_numerically(files):
        """
        Sort the file list based on numebers that are constantly changing.
        
        :param list(str) files: file list
        """
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

    def _key_to_uri(self, key):
        return os.path.join(self.root, key)
        
class ImageDatastore(FileDatastore):
    supported_extensions = ('tif', )

    def __init__(self, root, extensions=None, **kwargs):
        if extensions is None:
            extensions = ImageDatastore.supported_extensions
        super().__init__(
            root, extensions=extensions, **kwargs
        )
            