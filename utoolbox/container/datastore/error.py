class DatastoreError(Exception):
    """Base class for datastore exceptions."""

class InvalidDatastoreRootError(DatastoreError):
    """Unable to find the root directory."""

class ReadOnlyDataError(DatastoreError):
    """Datastore is readonly."""
    
class ImmutableUriListError(DatastoreError):
    """File list in datastore is immutable."""


class UnableToConvertError(DatastoreError):
    """Unable to convert from source datastore."""

class InvalidMetadataError(DatastoreError):
    """Invalid metadata in tar datastores."""
