from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Tuple

import xarray as xr


class Indexer(MutableMapping, ABC):
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

        self._index = None
        try:
            self._populate_index()
        except ValueError:
            # indexer has nothing to do with this object
            self._index = []
        else:
            # populate available index
            pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __iter__(self):
        for index in self._index:
            yield index

    def __len__(self):
        return len(self._index)

    ##

    @property
    @classmethod
    def allow_partial(self) -> bool:
        """Allow some of the suffixes combination missing."""
        return False

    @property
    def obj(self) -> xr.Dataset:
        """The xr.Dataset object to operate on."""
        return self._obj

    @property
    @classmethod
    @abstractmethod
    def coords(self) -> Tuple[str]:
        """Coordinate names require by this indexer."""

    @property
    @classmethod
    def attrs(self) -> Tuple[str]:
        """Attributes require by this indexer."""
        return tuple()

    ##

    def sel(self, **kwargs):
        # TODO
        pass

    ##

    def _populate_index(self):
        coords = []
        for coord in self.coords:
            if coord not in self.dataset.coords:
                if not self.allow_partial:
                    raise ValueError(f'missing coordinate "{coord}"')
            else:
                coords.append(coord)

        # save coordinate names that we can iterate over with
        self._iter_coords = coords



@xr.register_dataset_accessor("grid")
class GridIndexer(Indexer):
    coords = ("grid_x", "grid_y", "grid_z")
    allow_partial = True


@xr.register_dataset_accessor("coordinate")
class CoordinateIndexer(Indexer):
    coords = ("coord_x", "coord_y", "coord_z")
    allow_partial = True


@xr.register_dataset_accessor("view")
class ViewIndexer(Indexer):
    coords = ("view",)


@xr.register_dataset_accessor("time")
class TimeIndexer(Indexer):
    coords = ("time",)


@xr.register_dataset_accessor("channel")
class ChannelIndexer(Indexer):
    coords = ("channel",)
