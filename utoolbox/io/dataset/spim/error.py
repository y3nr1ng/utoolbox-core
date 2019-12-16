from ..base.error import DatasetError


class SpimDatasetError(DatasetError):
    """Generic SPIM dataset error."""


class MissingSettingsFileError(SpimDatasetError):
    """Unable to locate Settings.txt"""

class MissingScriptFileError(SpimDatasetError):
    """Unable to locate script file in CSV format."""