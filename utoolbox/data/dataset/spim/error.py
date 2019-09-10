from ..error import DatasetError


class SettingsError(DatasetError):
    """SPIM-generated settings related errors."""


class MultipleSettingsError(SettingsError):
    """Confuse between multiple settings."""


class SettingsNotFoundError(SettingsError):
    """Unable to find SPIM-generated settings."""


class SPIMPatternError(DatasetError):
    """SPIM-specific pattern error."""


class MissingFilenameComponentError(SPIMPatternError):
    """Filename componenet is missing."""
