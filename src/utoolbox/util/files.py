import math
import os
import logging

logger = logging.getLogger(__name__)

from PyQt5 import QtWidgets


class ExtensionFilter(object):
    def __init__(self, extensions):
        if isinstance(extensions, list):
            self._extensions = [e.lower() for e in extensions]
            self._judge = lambda x: all([x == e for e in self._extensions])
        else:
            self._extensions = extensions.lower()
            self._judge = lambda x: x == self._extensions

    def __call__(self, name):
        _, name = os.path.splitext(name)
        return self._judge(name[1:])


def list_files(root, name_filters=None):
    """List valid files under specific folder by filter conditions.

    Parameters
    ----------
    root : str
        Relative or absolute path that will be the root directory.
    name_filters : (optional) list of filters
        Filtering conditions.
    """
    file_list = []
    for name in os.listdir(root):
        if not name_filters or all([f(name) for f in name_filters]):
            file_list.append(os.path.join(root, name))
    logger.info('{} data found under "{}"'.format(len(file_list), root))
    return file_list


def get_local_directory(root=".", prompt="Select a directory..."):
    """Select an existing folder using system dialog."""
    return QtWidgets.QFileDialog.getExistingDirectory(None, prompt, root)


def get_open_file(root=".", prompt="Select a file..."):
    """Select an existing file using system dialog."""
    return QtWidgets.QFileDialog.getOpenFileName(None, prompt, root)[0]


def convert_size(size_bytes):
    """Convert bytes to human readable formatself."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])
