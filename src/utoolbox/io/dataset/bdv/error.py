from ..base.error import DatasetError


class BDVDatasetError(DatasetError):
    """Generic BigDataViewer dataset error."""

class InvalidChunkSizeError(BDVDatasetError):
    """Invalid chunk size."""