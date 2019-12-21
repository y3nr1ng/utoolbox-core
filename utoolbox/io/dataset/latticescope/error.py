from ..base.error import DatasetError


class LatticeScopeDatasetError(DatasetError):
    """Generic SPIM dataset error."""


class MissingSettingsFileError(LatticeScopeDatasetError):
    """Unable to locate Settings.txt"""


class MalformedSettingsFileError(LatticeScopeDatasetError):
    """Structural error in settings file."""
