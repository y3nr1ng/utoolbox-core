from ..base.error import DatasetError


class SmartSpimDatasteError(DatasetError):
    """Generic MM dataset error."""


class MissingMetadataError(SmartSpimDatasteError):
    """Unable to locate a metadata."""
