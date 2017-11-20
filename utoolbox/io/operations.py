from . import codec
from os.path import splitext

def imopen(name, **kwargs):
    """
    Create a closure to act as an image factory.
    """
    _FORMATS = {
        'tif': codec.tiff.Tiff
    }
    def _imobject(name):
        _, file_ext = splitext(name)
        file_ext = file_ext.strip('.').lower()
        return _FORMATS[file_ext]
    return _imobject(name)(name, **kwargs)
