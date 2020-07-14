import logging
from typing import Tuple

__all__ = ["Dataset"]

logger = logging.getLogger("utoolbox.io.dataset")

MODE_NAMES = {
    "r": "read-only",
    "r+": "read/write",
    "w": "single-volume",
    "w-": "multi-volume",
    "a": "any-mode",
}


class Dataset:
    """
    Args:
        name (str): short name of this format
        description (str): one-line description of the format
        modes (str list of str, optional): string or a list of string containing modes 
            that this format can handle
            - 'r', read-only, dateaset must exist
            - 'r+' for read/write, dateaset must exist
            - 'w', create dateaset, overwrite if exists
            - 'w-', create dataset, fail if exists
            - 'a', read/write if exists, create otherwise
    """

    def __init__(self, name, description, modes=None):
        self._name = name
        self._description = description

        self._modes = modes or "a"
        if isinstance(self._modes, str):
            self._modes = [self._modes]
        for mode in self._modes:
            # TODO validate modes
            pass
        self._modes = tuple(self._modes)

    def __repr__(self):
        # short description
        return f"<Dataset {self.name} - {self.description}"

    ##

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def modes(self) -> Tuple[str]:
        return self._modes

    ##

    def get_reader(self, request):
        pass

    def get_writer(self, request):
        pass
