import h5py

from .base import TransientDatastore

class HDF5Datastore(TransientDatastore):
    def __init__(self, root):
        """
        :param h5py.Dataset root: dataset handle
        """
        self._root = root

        # read-only
        super().__init__(
            read_func=lambda zt: self._root[zt, ...], 
            immutable=True
        )

    @property
    def filename(self):
        """File system location of the root HDF5 file."""
        return self._root.file.filename
