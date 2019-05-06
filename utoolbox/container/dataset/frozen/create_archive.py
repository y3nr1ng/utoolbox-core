from io import BytesIO
import json
import logging
import lzma 
from lzma import CHECK_NONE, FORMAT_XZ
import os
from pprint import pprint
import sys
from tarfile import TarFile, TarInfo

import coloredlogs
import imageio
from tqdm import tqdm
import xxhash

logger = logging.getLogger(__name__)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

def create_tar_datastore(root, overwrite=False):
    files = [fn for fn in os.listdir(root) if fn.endswith('.tif')]
    files.sort()
    logger.info("{} files found".format(len(files)))

    read_func = imageio.volread

    mode = 'w' if overwrite else 'x'
    tar = TarFile(name=os.path.basename(root), mode=mode)   

    filters = [
        {
            'id': lzma.FILTER_DELTA, 
            'dist': 5 
        },
        { 
            'id': lzma.FILTER_LZMA2,
            'dict_size': 2**24,
            'nice_len': 32
        },
    ]

    tarinfo = TarInfo()
    digests = []
    for fn in files:    
        print("[{}]".format(fn))

        fp = os.path.join(root, fn)
        data = read_func(fp)

        # serialization only works with byte array
        data = data.tobytes()
        print(".. {} bytes".format(sys.getsizeof(data)))

        digest = xxhash.xxh64_hexdigest(data)
        print(".. {}".format(digest))

        # TODO bsc compression
        #cdata = lzma.compress(data, filters=filters)
        cdata = data
        print(".. {} bytes (compressed)".format(sys.getsizeof(cdata)))

        fo = BytesIO(cdata)
        tarinfo.name = digest
        tarinfo.size = len(cdata)
        tar.addfile(tarinfo, fo)

    tar.close()

if __name__ == '__main__':
    src_dir = '~/Desktop/GH146ACV_power100_60ms_z3'
    src_dir = os.path.expanduser(src_dir)
    create_tar_datastore(src_dir, overwrite=True)