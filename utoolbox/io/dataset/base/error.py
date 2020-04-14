class DatasetError(Exception):
    """Generic dataset error."""


class UnsupportedDatasetError(DatasetError):
    """Dataset is in foreign format or malformed."""


class PreloadError(DatasetError):
    """Something wrong during preloading."""
