from abc import ABCMeta, abstractmethod
import logging

import xarray as xr

__all__ = ["GenericDataset"]

logger = logging.getLogger(__name__)


class GenericDataset(ABCMeta):
    def __init__(self):
        self._dataset = xr.Dataset()

    def __getattr__(self, attr):
        return getattr(self._dataset, attr)

    def __setattr__(self, attr, value):
        setattr(self._dataset, attr, value)

    ##

    def isel(self, indexers):
        pass

    def sel(self, indexers):
        pass

    def drop_sel(self, lbaels):
        pass

    ##

    def to_zarr(self):
        try:
            import zarr  # noqa
        except ImportError:
            logger.error("`zarr` has to be intsalled")

        # TODO collect axes info

    ##

    @property
    def attrs(self):
        """Dictioary of global attributes on this dataset."""
        pass

    @property
    def chunks(self):
        """Block dimensions for this datasets's data or None if it is not a dask array."""
        pass

    @property
    def coords(self):
        """Dictionary of xarray.DataArray objects corresponding to coordinate variables."""
        pass

    @property
    def data_vars(self):
        """Dictionary of xarray.DataArray objects corresponding to data variables."""
        pass

    @property
    def dims(self):
        """Mapping from dimension names to lengths."""
        pass

    @property
    def indexes(self):
        """Mapping of panda.Index objects used for label based indexing."""
        pass

    @property
    def loc(self):
        """Attribute for location based indexing."""
        pass

    @property
    def nbytes(self):
        """Actual size of the dataset."""
        pass

    @property
    def sizes(self):
        """Mapping from dimension names to lengths."""
        pass
