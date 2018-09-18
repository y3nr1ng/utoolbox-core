import logging
from operator import attrgetter
import os

from pprint import pprint

from . import Filename

__all__ = [
    'refactor_split_filenames'
]

logger = logging.getLogger(__name__)

def concat_timestamps(filenames):
    filenames[:] = [Filename(filename) for filename in filenames]
    filenames.sort(key=attrgetter('channel', 'name', 'stack'))

    ref_filename, name = None, None
    for filename in filenames:
        if ref_filename is None:
            ref_filename, name = filename, filename.name
        else:
            filename.name = name
            if ref_filename.channel == filename.channel:
                # update rest of the info for same channel only
                filename.stack = ref_filename.stack + 1
                filename.timestamp_rel = \
                    ref_filename.timestamp_rel + (filename.timestamp_abs-ref_filename.timestamp_abs)
            ref_filename = filename
    return filename

def refactor_split_filenames(filenames):
    pass
