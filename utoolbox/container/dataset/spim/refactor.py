import logging
import os

import tqdm

__all__ = ["merge_fragmented_timestamps", "rename_by_mapping"]

logger = logging.getLogger(__name__)

"""
if refactor:
    data_files_orig = copy.deepcopy(data_files)
    merge_fragmented_timestamps(data_files)
    rename_by_mapping(self.root, data_files_orig, data_files)
"""

# TODO create wrapper object for linked dataset


def rename_by_mapping(root, old_filenames, new_filenames):
    for old_fnobj, new_fnobj in tqdm(
        zip(old_filenames, new_filenames), total=len(new_filenames)
    ):
        old_fnstr = os.path.join(root, str(old_fnobj))
        new_fnstr = os.path.join(root, str(new_fnobj))
        if old_fnstr != new_fnstr:
            os.rename(old_fnstr, new_fnstr)


def merge_fragmented_timestamps(filenames, consolidate=False):
    """
    Merge timestamps from consecutive acquisitions. The file sequence is 
    assumed to be sorted accordingly!

    Parameters
    ----------
    TBA
    consolidate : bool
        TBA
    """
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
                assert (
                    filename.timestamp_abs > ref_filename.timestamp_abs
                ), "timestamp overflow occurred, no shit..."
                filename.timestamp_rel = ref_filename.timestamp_rel + (
                    filename.timestamp_abs - ref_filename.timestamp_abs
                )
            ref_filename = filename
