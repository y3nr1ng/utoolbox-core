class DatastoreError(Exception):
    """Base class for datastore exceptions."""

class InvalidMetadataError(DatastoreError):
    """Invalid metadata in tar datastores."""

class HashMismatchError(DatastoreError):
    """Digest mismatch after file decompression."""