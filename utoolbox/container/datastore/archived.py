import json
from lzma import FORMAT_XZ, LZMADecompressor
import tarfile

import xxhash

from .base import Datastore
from .error import HashMismatchError, InvalidMetadataError

class TarDatastore(Datastore):
    def __init__(self, root, read_func, memlimit=None):
        if not root.lower().endswith('.tar'):
            raise ValueError("not a TAR file")   
        self._root = root

        tar = tarfile.open(self.root, 'r:')
        self._tar = tar
        self._inventory = tar.getmembers()

        #TODO do we need metadata?
        try:
            tarinfo = tar.getmember('metadata')
            metadata = tar.extractfile(tarinfo).read()
            self._metadata = json.load(metadata)
        except KeyError:
            raise InvalidMetadataError()

        self._decompressor = LZMADecompressor(
            format=FORMAT_XZ, memlimit=memlimit
        )
        def _wrapped_read_func(tarinfo):
            """Read from compressed data block."""
            data = self._tar.extractfile(tarinfo).read()
            #TODO decompression
            #data = self._decompressor.decompress(data)

            digest = xxhash.xxh64_hexdigest(data)
            if tarinfo.name != digest:
                raise HashMismatchError("hash mismatch after decompression")

            return read_func(data)
        super(TarDatastore, self).__init__(_wrapped_read_func)

        # non-wrapped read_func
        self._base_read_func = read_func
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._tar.close()

    @property
    def root(self):
        return self._root
