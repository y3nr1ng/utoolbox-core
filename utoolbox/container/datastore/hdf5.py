import h5py

from .base import Datastore

class HDF5Datastore(Datastore):
    def __init__(self, root):
        if isinstance(root, h5py.File):
            self._handle = root
        else:
            self._handle = h5py.File(root, 'r')

        self._inventory = self.handle.keys()
        super().__init__(lambda x: self.handle[x])
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()

    @property
    def handle(self):
        return self._handle

    @property
    def root(self):
        """Location of the HDF5 file."""
        return self.handle.filename
