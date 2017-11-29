import numpy as np

class SimpleVolume(np.ndarray):
    def __new__(subtype, *args, **kwargs):
        # create SimpleVolume with the usual ndarray input arguments
        # NOTE SimpleVolume.__array_finalize__ is triggered
        obj = super(SimpleVolume, subtype).__new__(subtype, args, kwargs)
        #TODO set new attributes
        return obj

    def __array_finalize__(self, obj):
        """
        `self` is a new object resulting from ndarray.__new__(SimpleVolume, ...)
        Therefore, it only has attributes that the ndarray.__new__ constructor
        gave it.
        """
        # from explicit constructor
        #   in the middle of SimpleVolume.__new__ construcotr
        if obj is None:
            return

        # from view casting
        #   obj is an array, type(obj) can be SimpleVolume

        # from new-from-template, such as slicing
        #   type(obj) is SimpleVolume

        #TODO do additional processing if required
    
    def __init__(self, path):
        self._path = path

    def __del__(self):
        pass

class SIVolume(object):
    pass
