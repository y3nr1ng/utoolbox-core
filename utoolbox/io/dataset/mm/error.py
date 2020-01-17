from ..base.error import DatasetError


class MicroManagerDatasetError(DatasetError):
    """Generic MM dataset error."""


class MissingMetadataError(MicroManagerDatasetError):
    """Unable to locate a metadata."""


class MalformedMetadataError(MicroManagerDatasetError):
    """Metadata file structure is corrupted."""
