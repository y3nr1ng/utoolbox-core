class DatasetError(Exception):
    """Base class for dataset exceptions."""

class UndefinedConversionError(DatasetError):
    """Raised when dataset conversion is impossible."""