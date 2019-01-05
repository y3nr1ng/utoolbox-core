# pylint: disable=W1401

import logging
import os
import re

from utoolbox.container import Datastore
from .dataset import Dataset

__all__ = [
    'Filename',
    'sort_by_timestamp'
]

logger = logging.getLogger(__name__)

class Filename(object):
    __slots__ = (
        'name', 'channel', 'stack', 'wavelength', 'timestamp_rel', 'timestamp_abs'
    )

    _pattern = re.compile(
        r"(?P<name>\w+)_"
        r"ch(?P<channel>\d+)_"
        r"stack(?P<stack>\d{4})_"
        r"(?P<wavelength>\d+)nm_"
        r"(?P<timestamp_rel>\d{7})msec_"
        r"(?P<timestamp_abs>\d+)msecAbs"
        r".tif{1,2}$"
    )

    def __init__(self, str_name):
        parsed_name = Filename._pattern.fullmatch(str_name)
        if not parsed_name:
            raise ValueError
        for attr in self.__slots__:
            value = parsed_name.group(attr)
            value = value if attr == 'name' else int(value)
            setattr(self, attr, value)

    def __str__(self):
        return "{}_ch{}_stack{:04d}_{}nm_{:07d}msec_{:010d}msecAbs.tif" \
               .format(*[getattr(self, attr) for attr in self.__slots__])

def _to_filename_objs(ds):
    """
    Convert a datastore file list to list of filename objects.

    Parameter
    ---------
    ds : utoolbox.container.Datastore
        Datastore to process.
    """
    filenames = []
    for filename in ds.files:
        basename = os.path.basename(filename)
        try:
            parsed = Filename(basename)
            filenames.append(parsed)
        except:
            logger.warning("invalid format \"{}\", ignored".format(basename))
    return filenames

def _sort_datastore_by_timestamp(ds):
    filename_objs = _to_filename_objs(ds)
    ds.files = [
        path for _, path in sorted(
            zip(filename_objs, ds.files), 
            key=lambda t: (t[0].channel, t[0].name, t[0].stack)
        )
    ]
    
def sort_by_timestamp(source):
    print(type(source))
    if isinstance(source, Dataset):
        # iterate over different channels
        for datastore in source.datastore.values():
            _sort_datastore_by_timestamp(datastore)
    else:
        _sort_datastore_by_timestamp(source)