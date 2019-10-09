from collections import OrderedDict
import re

__all__ = ["AttrDict"]


class AttrDict(OrderedDict):
    """
    Metadata provides the means to get and set keys as attributes while behaves
    as much as possible as a normal dict. Keys that are not valid identifiers or
    names of keywords cannot be used as attributes.

    Reference: 
        - imageio.core.util.Dict
    """

    __reserved_names__ = dir(OrderedDict())
    __pure_names__ = dir(dict())

    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as err:
            try:
                return self[key]
            except KeyError:
                # override by original error
                raise err

    def __setattr__(self, key, val):
        if key in AttrDict.__reserved_names__:
            if key not in AttrDict.__pure_names__:
                return OrderedDict.__setattr__(self, key, val)
            else:
                raise AttributeError(
                    "reserved name can only be set via dictionary interface".format(key)
                )
        else:
            self[key] = val

    def __dir__(self):
        def is_identifier(x):
            return bool(re.match(r"[a-z_]\w*$", x, re.I))

        names = [k for k in self.keys() if isinstance(k, str) and is_identifier(k)]
        return AttrDict.__reserved_names__ + names
