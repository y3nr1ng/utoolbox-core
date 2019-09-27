import h5py

from utoolbox.data.datastore import FileDatastore, TransientDatastore


class HDF5Datastore(FileDatastore, TransientDatastore):
    """
    Args:
        path (str): 
    """

    def __init__(self, path):
        """
        :param h5py.Dataset root: dataset handle`
        """
        self._root = root

        # read-only
        super().__init__(read_func=lambda zt: self._root[zt, ...], immutable=True)

    def _allocate_resources(self):
        pass

    def _free_resources(self):
        pass

    @property
    def filename(self):
        """File system location of the root HDF5 file."""
        return self._root.file.filename
