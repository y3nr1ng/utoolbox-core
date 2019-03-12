import logging
import os
from typing import NamedTuple

try:
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False
else:
    USE_TQDM = True

from .base import Datastore

logger = logging.getLogger(__name__)
if USE_TQDM:
    from utoolbox.util.logging import TqdmLoggingHandler
    logger.addHandler(TqdmLoggingHandler())

class DatastoreDescriptor(NamedTuple):
    ds_type: Datastore
    uri: str = ''
    pattern: str = '*'
    extension: str = 'tif'
    read_func: object = None
    write_func: object = None

def convert(dst, src):
    #TODO validate the data

    with src.ds_type(src.uri, src.read_func, pattern=src.pattern) as src_ds:
        # create target directory
        dst_uri = dst.uri
        if not dst_uri:
            parent = os.path.dirname(src.uri)
            ds_name = os.path.basename(src.uri)
            ds_name = '{}_converted'.format(ds_name)
            dst_uri = os.path.join(parent, ds_name)
            logger.info("default destination \"{}\"".format(dst_uri))
        os.makedirs(dst_uri)

        dst_extension = dst.extension if dst.extension else src.extension

        iterator = zip(src_ds.files, src_ds)
        if USE_TQDM:
            iterator = tqdm(iterator, total=len(src_ds))
        for fn, data in iterator:
            fn = os.path.basename(fn)
            fp = os.path.join(dst_uri, "{}.{}".format(fn, dst_extension))
            dst.write_func(fp, data)