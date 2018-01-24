import logging
logger = logging.getLogger(__name__)

import os.path

import numpy as np

from utoolbox.io.operations import imopen

class SimpleVolume(np.ndarray):
    def __new__(subtype, file_path=None, shape=None):
        """
        Create ndarray instance of SimpleVolume, given the usual arguments.

        Parameters
        ----------
        file_path: string
            path to the image
        shape: tuple of integers
            shape of the volumetric data

        Returns
        -------
        SimpleVolume
            newly created volume object, child class of np.ndarray
        """
        if file_path:
            if os.path.exists(file_path):
                # read from file
                with imopen(file_path, 'r') as imfile:
                    shape = imfile.shape
                    dtype = imfile.dtype
                    obj = super(SimpleVolume, subtype).__new__(subtype,
                                                               shape=shape,
                                                               dtype=dtype)
                    for index, page in enumerate(imfile):
                        obj[..., index] = page.raster
            elif shape:
                raise NotImplementedError("Create new file is not implemented.")
                obj = super(SimpleVolume, subtype).__new__(subtype,
                                                           shape=shape)
            else:
                raise ValueError("'shape' must be provided to allocate spaces.")
        else:
            raise ValueError("'file_path' must be provided.")

        obj._path = file_path
        return obj

    def __array_finalize__(self, obj):
        """
        `self` is a new object resulting from ndarray.__new__(SimpleVolume, ...)
        Therefore, it only has attributes that the ndarray.__new__ constructor
        gave it.
        """
        # from explicit constructor
        #   in the middle of SimpleVolume.__new__ constructor
        if obj is None:
            #TODO load data here
            print('obj is None')
            return

        # from view casting
        #   obj is an array, type(obj) can be SimpleVolume


        # from new-from-template, such as slicing
        #   type(obj) is SimpleVolume


    def __init__(self, path):
        self._path = path

    def __del__(self):
        pass

class SIVolume(object):
    pass
