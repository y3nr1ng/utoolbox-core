import logging
import os

from utoolbox.container import AbstractDataset

logger = logging.getLogger(__name__)

class FrozenDataset(AbstractDataset):
    """
    Representation of a frozen dataset in compressed TAR format.
    """
    def __init__(self, root):
        if not root.lower().endswith('.tar'):
            raise ValueError("not in frozen format")
        super().__init__(root)

        #TODO read_func attach with dynamic decompression

    @staticmethod
    def from_dataset(ds, **kwargs):
        pass
        
        #TODO create tar file
        filename = os.path.basename(ds.root)

        #TODO save inventory

        for channel in ds:
            

        #TODO iterate over files 
        #TODO .. compress in lzma (xz)
        #TODO .. write chunk
        #TODO .. xxhash

        #TODO consolidate tar
    
    @staticmethod
    def _from_spimdataset(ds):
        pass
    
    @staticmethod
    def _from_genericdataset(ds):
        pass

    def _generate_inventory(self):
        raise NotImplementedError
    
    def _load_datastore(self):
        raise NotImplementedError