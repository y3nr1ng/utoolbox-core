class DatastoreError(Exception):
    """Base class for datastore exceptions."""

class UnableToConvertError(DatastoreError):
    """Unable to convert from source datastore."""

class InvalidMetadataError(DatastoreError):
    """Invalid metadata in tar datastores."""
