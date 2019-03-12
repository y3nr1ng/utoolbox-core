"""
Datastores that can direct access the files.
"""
import glob
import logging
import os

from . import SparseStackImageDatastore
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
        self._inventory = files

    @property
    def files(self):
        return self._inventory

class ImageDatastore(FileDatastore):
    supported_extensions = ['tif']

    def __init__(self, root, read_func, extensions=None, **kwargs):
        if extensions is None:
            extensions = ImageDatastore.supported_extensions
        super().__init__(
            root, read_func=read_func, extensions=extensions, **kwargs
        )

    @staticmethod
    def convert_from(dst, src, write_func):
        if src is SparseStackImageDatastore:
            ImageDatastore._from_sparse_stack(dst, src, write_func)
        else:
            super().convert_from(src)

    @staticmethod
    def _from_sparse_stack(dst, src, write_func, extension='tif'):
        os.mkdirs(dst)
        for fn, im in zip(src.files, src):
            fp = os.path.join(dst, "{}.{}".format(fn, extension))
            write_func(fp, im)
            