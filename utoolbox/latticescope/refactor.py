import logging
from operator import attrgetter
import os

from .parse import Filename

__all__ = [
    'sort_timestamps',
    'merge_fragmented_timestamps',
    'rename_by_mapping'
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
    #TODO add progress indicator
    for old_fnobj, new_fnobj in zip(old_filenames, new_filenames):
        old_fnstr = os.path.join(root, str(old_fnobj))
        new_fnstr = os.path.join(root, str(new_fnobj))
        if old_fnstr != new_fnstr:
            os.rename(old_fnstr, new_fnstr)
