import logging
from operator import attrgetter
import os

from .parse import Filename

__all__ = [
    'sort_timestamps',
    'merge_fragmented_timestamps'
]

logger = logging.getLogger(__name__)

def sort_timestamps(filenames):
    filenames.sort(key=attrgetter('channel', 'name', 'stack'))

def merge_fragmented_timestamps(filenames):
    ref_filename, name = None, None
    for filename in filenames:
        if ref_filename is None:
            ref_filename, name = filename, filename.name
        else:
            filename.name = name
            if ref_filename.channel == filename.channel:
                # update rest of the info for same channel only
                if filename.stack == 0:
                    logger.debug("rewind")
                filename.stack = ref_filename.stack + 1
                assert filename.timestamp_abs > ref_filename.timestamp_abs, \
                       "timestamp overflow occurred, no shit..."
                filename.timestamp_rel = \
                    ref_filename.timestamp_rel + (filename.timestamp_abs-ref_filename.timestamp_abs)
            ref_filename = filename

def rename_by_mapping(root, old_filenames, new_filenames):
    for old_filename, new_filename in zip(old_filenames, new_filenames):
        old_filename = os.path.join(root, str(old_filename))
        new_filename = os.path.join(root, str(new_filename))
        if old_filename != new_filename:
            os.rename(old_filename, new_filename)
