from ..base.error import DatasetError


class LatticeScopeDatasetError(DatasetError):
    """Generic SPIM dataset error."""


class MissingSettingsFileError(LatticeScopeDatasetError):
    """Unable to locate Settings.txt"""

class MissingScriptFileError(LatticeScopeDatasetError):
    """Unable to locate script file in CSV format."""