from collections import OrderedDict
import copy
import logging
import re


__all__ = ["refactor_datastore_keys"]

logger = logging.getLogger(__name__)


class Filename(object):
    __slots__ = (
        "name",
        "channel",
        "stack",
        "wavelength",
        "timestamp_rel",
        "timestamp_abs",
    )

    _pattern = re.compile(
        r"(?P<name>\w+)_"
        r"ch(?P<channel>\d+)_"
        r"stack(?P<stack>\d{4})_"
        r"(?P<wavelength>\d+)nm_"
        r"(?P<timestamp_rel>\d{7})msec_"
        r"(?P<timestamp_abs>\d+)msecAbs$"
    )

    def __init__(self, str_name):
        parsed_name = Filename._pattern.fullmatch(str_name)
        if not parsed_name:
            raise ValueError
        for attr in self.__slots__:
            value = parsed_name.group(attr)
            value = value if attr == "name" else int(value)
            setattr(self, attr, value)

    def __str__(self):
        return "{}_ch{}_stack{:04d}_{}nm_{:07d}msec_{:010d}msecAbs".format(
            *[getattr(self, attr) for attr in self.__slots__]
        )


def keys_to_filename_objs(keys):
    """
    Convert a datastore key list to list of filename objects.
    """
    filenames = []
    for key in keys:
        try:
            parsed = Filename(key)
            filenames.append(parsed)
        except ValueError:
            logger.warning(f'invalid format "{key}", ignored')
    return filenames


def refactor_datastore_keys(datastore):
    old_fno = keys_to_filename_objs(datastore.keys())
    old_fno.sort(key=lambda fno: (fno.channel, fno.name, fno.stack))
    logger.info("found {} entries to refactor".format(len(old_fno)))

    new_fno = copy.deepcopy(old_fno)
    merge_fragmented_timestamps(new_fno)

    logger.info("remapping refactored keys")
    new_uri = OrderedDict()
    for new, old in zip(new_fno, old_fno):
        new_uri[str(new)] = datastore._uri[str(old)]

    # overwrite
    datastore._uri = new_uri


def merge_fragmented_timestamps(filenames):
    """
    Merge timestamps from consecutive acquisitions. The file sequence is 
    assumed to be sorted accordingly!
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
