import logging
logger = logging.getLogger(__name__)

import numpy as np
import imageio

from .registry import BaseContainer

class Rasters(BaseContainer, np.ndarray):
    """Container for data represented by dense arrays.

    Reference
    ---------
    imageio.core.util.Image
    """
    def __new__(cls, source=None, **kwargs):
        if source is None:
            obj = np.ndarray.__new__(cls, **kwargs)
        else:
            if isinstance(source, np.ndarray):
                obj = source.view(cls)
            else:
                obj = cls._load_externally(source).view(cls)

        return obj

    def __array_finalize__(self, obj):
        if isinstance(obj, Rasters):
            # from view-casting
            self._copy_metadata(obj.metadata)
        else:
            # in the middle of __new__ or from templating
            if obj is not None:
                self.metadata.resolution = tuple([1.] * self.ndim)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # take note which inputs are converted to ndarray
        args = []
        input_no = []
        for i, arg in enumerate(inputs):
            if isinstance(arg, Rasters):
                input_no.append(i)
                args.append(arg.view(np.ndarray))
            else:
                args.append(arg)
        logger.debug("native input @ {}".format(input_no))

        outputs = kwargs.pop('out', None)
        output_no = []
        if outputs:
            out_args = []
            for i, arg in enumerate(outputs):
                if isinstance(arg, Rasters):
                    output_no.append(i)
                    out_args.append(arg.view(np.ndarray))
                else:
                    out_args.append(arg)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None, ) * ufunc.nout
        logger.debug("native output @ {}".format(input_no))

        # run the actual ufunc
        results = np.ndarray.__array_ufunc__(
            self, ufunc, method, *args, **kwargs
        )
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results, )

        results = tuple(
            (np.asarray(result).view(Rasters) if output is None else output)
            for result, output in zip(results, outputs)
        )

        if method == 'reduce':
            axis = kwargs.get('axis', None)
            if axis is None:
                axis = ()
            elif not isinstance(axis, tuple):
                axis = (axis, )

            # cherry-pick the resolution
            if not kwargs.get('keepdims', False):
                if axis:
                    resolution = tuple(
                        x for i, x in enumerate(inputs[0].metadata.resolution)
                        if i not in axis
                    )
                else:
                    # unit spacing for scalar
                    resolution = (1., )
                results[0].metadata.resolution = resolution

        return results[0] if len(results) == 1 else results

    @staticmethod
    def _load_externally(source):
        raise NotImplementedError

class Image(Rasters):
    """2-D, single channel image."""
    @staticmethod
    def _load_externally(source):
        #TODO use utoolbox.io to determine the proper way to open
        return imageio.imread(source)

class Volume(Rasters):
    """3-D, single channel image."""
    @staticmethod
    def _load_externally(source):
        #TODO use utoolbox.io to determine the proper way to open
        return imageio.volread(source)
