"""
Datastore is a respository for collections of data that may be too large to fit in memory, allowing one to read and process data stored in multiple files on a disk. 

.. note::
    Datastore imitates and expands upon the MATLAB equivalent.
"""
from .archived import *
from .convert import convert, DatastoreDescriptor
from .direct import *
from .sparse import *