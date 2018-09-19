import re

__all__ = [
    'Filename'
]

class Filename(object):
    __slots__ = (
        'name', 'channel', 'stack', 'wavelength', 'timestamp_rel', 'timestamp_abs'
    )

    _pattern = re.compile(
        "(?P<name>\w+)_"
        "ch(?P<channel>\d+)_"
        "stack(?P<stack>\d{4})_"
        "(?P<wavelength>\d+)nm_"
        "(?P<timestamp_rel>\d{7})msec_"
        "(?P<timestamp_abs>\d+)msecAbs"
        ".tif{1,2}$"
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
