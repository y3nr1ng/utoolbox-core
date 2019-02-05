
from utoolbox.container import AbstractDataset

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
    def convert_from(ds):
        #TODO create tar file

        #TODO save inventory

        #TODO iterate over files 
        #TODO .. compress in lzma (xz)
        #TODO .. write chunk
        #TODO .. xxhash

        #TODO consolidate tar
        pass
    
    def preview(self, view='all'):
        raise NotImplementedError
    
    def _generate_inventory(self):
        raise NotImplementedError
    
    def _load_datastore(self):
        raise NotImplementedError