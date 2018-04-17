import logging
logger = logging.getLogger(__name__)

import numpy as np
import imageio

from .registry import BaseContainer

class Raster(BaseContainer, np.ndarray):
    """Container for data represented by dense arrays.

    Parameters
    ----------
    layout : utoolbox.io.layouts
        Type of the raster.
    """
    def __new__(cls, layout, source=None, **kwargs):
        if source is None:
            obj = np.ndarray.__new__(cls, **kwargs)
        else:
            if not isinstance(source, np.ndarray):
                source = layout.read(source)
            obj = source.view(cls)

        obj.layout = layout

        return obj

    def __array_finalize__(self, obj):
        if isinstance(obj, Raster):
            # from view-casting
            self._copy_metadata(obj.metadata)
        else:
            # in the middle of __new__ or from templating
            if obj is not None:
                self.metadata.resolution = tuple([1.] * self.ndim)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # convert inputs to ndarray
        in_args = []
        in_ind = []
        for i, arg in enumerate(inputs):
            if isinstance(arg, Raster):
                in_ind.append(i)
                in_args.append(arg.view(np.ndarray))
            else:
                in_args.append(arg)
        logger.debug("{} inputs require conversion".format(len(in_ind)))
        if in_ind:
            logger.debug("... {}".format(in_ind))

        out_args = kwargs.pop('out', None)
        out_ind = []
        if out_args:
            args = []
            for i, arg in enumerate(out_args):
                if isinstance(arg, Raster):
                    out_ind.append(i)
                    args.append(arg.view(np.ndarray))
                else:
                    args.append(arg)
            kwargs['out'] = tuple(args)
        else:
            out_args = (None, ) * ufunc.nout
        logger.debug("{} outputs require conversion".format(len(out_ind)))
        if out_ind:
            logger.debug("... {}".format(out_ind))
        out_ind = tuple(out_ind)

        # run the actual ufunc
        results = np.ndarray.__array_ufunc__(
            self, ufunc, method, *in_args, **kwargs
        )
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results, )

        results = tuple(
            (np.asarray(result).view(Raster) if arg is None else arg)
            for result, arg in zip(results, out_args)
        )

        logger.debug(results)

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

    def save(self, file):
        self.layout.write(file, self)
