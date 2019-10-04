"""
Datastores that can direct access the files.
"""
from abc import abstractmethod
import glob
import logging
import os
import re

from .base import Datastore
from .error import InvalidDatastoreRootError

logger = logging.getLogger(__name__)

__all__ = ["FileDatastore", "FolderDatastore", "ImageFolderDatastore"]


class FileDatastore(Datastore):
    """
    A datastore represents by a single file, such as HDF5.

    Args:
        path (str): path to the file
        create_new (bool): create the file if not exists
    """

    def __init__(self, path, create_new=False, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def _enumerate_entries(self):
        """Enumerate what are stored in this file."""


class FolderDatastore(Datastore):
    """
    A datastore represents by a folder that contains numerous files.
    
    Args:
        root (str): root folder of the datastore
        sub_dir (bool): scan nested folders for files
        pattern (str): filename pattern
        extensions (:obj:`list` of str): file extensions to include
        create_new (bool): create the root folder if not exists
    """

    def __init__(
        self,
        root,
        sub_dir=False,
        pattern="*",
        extensions=None,
        create_new=True,
        **kwargs,
    ):
        if "del_func" not in kwargs:
            kwargs["del_func"] = os.unlink

        super().__init__(**kwargs)

        if not os.path.exists(root):
            if create_new:
                os.mkdir(root)
            else:
                raise InvalidDatastoreRootError('unable to find "{}"'.format(root))
        self._root = root

        if sub_dir:
            root = os.path.join(root, "**")
        logger.debug('search under "{}"'.format(root))

        if extensions is None:
            extensions = [pattern]
        else:
            extensions = ["{}.{}".format(pattern, ext) for ext in extensions]
        logger.debug("{} search patterns".format(len(extensions)))

        files = []
        for ext in extensions:
            path = os.path.join(root, ext)
            files.extend(glob.glob(path, recursive=sub_dir))
        FolderDatastore._sort_numerically(files)

        # simple 1-1 mapping
        for path in files:
            key = os.path.basename(path)
            key, _ = os.path.splitext(key)
            self._uri[key] = path

    @property
    def root(self):
        return self._root

    @staticmethod
    def _sort_numerically(files):
        """
        Sort the file list based on numebers that are constantly changing.
        
        Args:
            files (:obj:`list` of str): file list
        """
        # extract valid numbers from filenames
        keys = [list(map(int, re.findall(r"[0-9]+", fn))) for fn in files]
        # identify constant variables
        flags = [all(elem == elems[0] for elem in elems) for elems in zip(*keys)]
        # keep varying keys
        keys[:] = [[k for k, f in zip(key, flags) if not f] for key in keys]

        # sort the file list based on extracted keys
        files[:] = [f for _, f in sorted(zip(keys, files))]

    def _key_to_uri(self, key):
        """Convert key to URI when writing new data into datastore."""
        return os.path.join(self.root, key)


class ImageFolderDatastore(FolderDatastore):
    supported_extensions = ("tif",)

    def __init__(self, root, **kwargs):
        if "extensions" not in kwargs:
            kwargs["extensions"] = ImageFolderDatastore.supported_extensions
        super().__init__(root, **kwargs)

