import logging

import imageio
import itk
import numpy as np
import SimpleITK as sitk

from .base import BaseContainer

logger = logging.getLogger(__name__)

class Raster(BaseContainer, np.ndarray):
    """Container for data represented by dense arrays.
    """
    def __new__(cls, array=None, **kwargs):
        layout = kwargs.pop('layout', None)
        if array is None:
            obj = np.ndarray.__new__(cls, **kwargs)
        else:
            if not isinstance(array, cls):
                try:
                    array = layout.read(array)
                except AttributeError:
                    raise TypeError("logical layout not specified")
            #TODO import from itk
            obj = array.view(cls)
        obj.metadata.layout = layout
        return obj

    def __str__(self):
        size = 'x'.join([str(i) for i in self.shape])
        return "Raster, {}, {}".format(size, self.dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # convert input of Raster to ndarray
        in_args = []
        for _in in inputs:
            if isinstance(_in, Raster):
                in_args.append(_in.view(np.ndarray))
            else:
                in_args.append(_in)

        # temporary convert output of Raster to ndarray
        if 'out' in kwargs:
            out_args = []
            for _out in kwargs.pop('out', None):
                if isinstance(_out, Raster):
                    out_args.append(_out.view(np.ndarray))
                else:
                    out_args.append(_out)
            kwargs['out'] = tuple(out_args)
        else:
            out_args = (None, ) * ufunc.nout

        # run the method
        results = np.ndarray.__array_ufunc__(
            self, ufunc, method, *in_args, **kwargs
        )
        if results is NotImplemented:
            return NotImplemented
        if ufunc.nout == 1:
            results = (results, )

        # convert back to Raster and duplicate metadata
        outputs = tuple(
            (np.asarray(res).view(Raster) if _out is None else _out)
            for res, _out in zip(results, out_args)
        )
        for _out in outputs:
            if isinstance(_out, Raster):
                _out._copy_metadata(self.metadata)

        # trim unused result
        return outputs[0] if len(outputs) == 1 else outputs

    def save(self, dst):
        self.metadata.layout.write(dst, self)

    def to_sitk(self):
        """
        Convert to SimpleITK image container format with relevant parameters.
        """
        image = sitk.GetImageFromArray(self)

        # migrate spacing if assigned
        try:
            spacing = list(self.metadata.spacing)
        except AttributeError:
            spacing = [1] * self.ndim
        # sitk used reversed ordering
        image.SetSpacing(spacing[::-1])

        return image

    def to_itk(self):
        """
        Convert to ITK image container format with relevant parameters.
        """
        image = itk.GetImageFromArray(self)

        filter = itk.ChangeInformationImageFilter[image].New()
        filter.SetInput(image)

        # migrate spacing if assigned
        try:
            spacing = list(self.metadata.spacing)
        except AttributeError:
            spacing = [1] * self.ndim
        # itk used reversed ordering
        filter.SetOutputSpacing(spacing[::-1])
        filter.ChangeSpacingOn()

        filter.UpdateOutputInformation();

        return filter.GetOutput()
