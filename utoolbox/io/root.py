"""
Guess where the data directory is on this computer.
"""


# retrieve module path
import inspect
import os

import utoolbox

module_path = inspect.getmodule(utoolbox).__path__._path[0]
DATA_ROOT = os.path.join(module_path, 'data')
