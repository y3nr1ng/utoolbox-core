import abc
from collections import OrderedDict
import logging
import re

logger = logging.getLogger(__name__)

class Metadata(OrderedDict):
    """
    Metadata provides the means to get and set keys as attributes while behaves
    as much as possible as a normal dict. Keys that are not valid identifiers or
    names of keywords cannot be used as attributes.

    Reference
    ---------
    imageio.core.util.Dict
    """
    __reserved_names__ = dir(OrderedDict())
    __pure_names__ = dir(dict())

    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key in self:
                return self[key]
            else:
                raise

    def __setattr__(self, key, val):
        if key in Metadata.__reserved_names__:
            if key not in Metadata.__pure_names__:
                return OrderedDict.__setattr__(self, key, val)
            else:
                raise AttributeError(
                    "reserved name can only be set via `metadata[{}] = X`" \
                    .format(key)
                )
        else:
            self[key] = val

    def __dir__(self):
        is_identifier = lambda x: bool(re.match(r'[a-z_]\w*$', x, re.I))
        names = [
            k for k in self.keys() if isinstance(k, str) and is_identifier(k)
        ]
        return Metadata.__reserved_names__ + names

class BaseContainer(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        metadata : dict, optional
            Metadata to attach with this container.
        resolution : tuple or list, optional
            Resolution of the unit elements in the container. Default to
            isotropic spacing between dimensions.
        """
        self._copy_metadata(kwargs.pop('metadata', {}))
        self._set_metadata(**kwargs)

    def _copy_metadata(self, metadata):
        """Make a 2-level deep copy of the metadata dictionary."""
        for key, val in metadata.items():
            if isinstance(val, dict):
                val = Metadata(val)
            self.metadata[key] = val

    def _set_metadata(self, **kwargs):
        """Save directly assigned attributes as part of the metadata."""
        for key in list(kwargs):
            self.metadata[key] = kwargs.pop(key)

    @property
    def metadata(self):
        try:
            return self._metadata
        except AttributeError:
            self._metadata = Metadata()
            return self._metadata

    @abc.abstractmethod
    def save(self, dst):
        pass
