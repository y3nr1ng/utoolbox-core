import h5py

from utoolbox.data.datastore import FileDatastore, TransientDatastore


class HDF5Datastore(FileDatastore, TransientDatastore):
    """
    Args:
        root (str or h5py.Dataset): file path or dataset handle 
    """

    def __init__(self, root, *args, **kwargs):
        self._root = root

        # TODO read-only
        super().__init__(immutable=True)

    ##

    def _allocate_resources(self):
        self._handle = h5py.File(self._root)

    def _free_resources(self):
        self.handle.close()

    ##

    def _enumerate_entries(self):
        pass

    ##

    @property
    def filename(self):
        """File system location of the root HDF5 file."""
        try:
            return self._root.file.filename
        except AttributeError:
            # file not opened yet
            return self._root

    @property
    def handle(self):
        return self._handle
