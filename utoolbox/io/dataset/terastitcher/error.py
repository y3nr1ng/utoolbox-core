from ..base.error import DatasetError


class TeraStitcherDatasetError(DatasetError):
    """Generic MM dataset error."""


class MissingMetadataError(TeraStitcherDatasetError):
    """Unable to locate a metadata."""
