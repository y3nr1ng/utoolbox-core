class DatastoreError(Exception):
    """Base class for datastore exceptions."""

class ReadOnlyDataError(DatastoreError):
    """Data in current datastore is immutable."""

class ImmutableUriListError(DatastoreError):
    """File list in datastore is immutable."""

class UnableToConvertError(DatastoreError):
    """Unable to convert from source datastore."""

class InvalidMetadataError(DatastoreError):
    """Invalid metadata in tar datastores."""
