from functools import partial

import coloredlogs
from imageio import imread, volwrite

from utoolbox.container.datastore import (
    convert,
    DatastoreDescriptor,
    ImageFolderDatastore,
    SparseStackImageFolderDatastore,
)

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s  %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

src = DatastoreDescriptor(
    ds_type=SparseStackImageFolderDatastore,
    uri="/home/ytliu/nas/hive_archive/ytliu/brain_clarity_lectin/poststain_640_z5",
    pattern="*_640_*",
    read_func=imread,
)

volwrite_ = partial(volwrite, bigtiff=True, software="utoolbox")
dst = DatastoreDescriptor(
    ds_type=ImageFolderDatastore,
    uri="/home/ytliu/nas/hive_buffer/ytliu/brain_clarity_lectin/poststain_640_z5_dense",
    write_func=volwrite_,
)

convert(dst, src)
