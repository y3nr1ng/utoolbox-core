from ..error import DatasetError


class StructureError(DatasetError):
    """Unexpected Micro-Manager folder layout."""


class NoMetadataInTileFolderError(StructureError):
    """Cannot find metadata.txt in subfolders."""


class MetadataError(DatasetError):
    """Metadata file format error."""


class NoSummarySectionError(MetadataError):
    """Unable to find summary section."""
