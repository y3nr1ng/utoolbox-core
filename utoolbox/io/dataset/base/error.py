class DatasetError(Exception):
    """Generic dataset error."""


class UnsupportedDatasetError(DatasetError, TypeError):
    """Dataset is in foreign format or malformed."""
