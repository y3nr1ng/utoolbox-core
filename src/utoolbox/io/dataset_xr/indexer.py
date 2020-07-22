from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Tuple

from .dataset import Dataset


class Indexer(MutableMapping, ABC):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._validate_requirements()

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

    ##

    @property
    def dataset(self) -> Dataset:
        """Parent dataset this indexer points to."""
        return self._dataset

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

    def loc(self, **kwargs):
        pass

    ##

    def _validate_requirements(self):
        self._validate_coords()
        self._validate_attrs()

    def _validate_coords(self):
        for coord in self.coords:
            if coord not in self.dataset.coords:
                raise ValueError(f'missing coordinate "{coord}"')

    def _validate_attrs(self):
        for attr in self.attrs:
            if attr not in self.dataset.attrs:
                raise ValueError(f'missing attribute "{attr}"')


class GroupedIndexer(Indexer):
    """
    Indexer that iterate through a group of coordinates at once.
    """

    @property
    @classmethod
    @abstractmethod
    def coord_suffixes(self) -> Tuple[str]:
        """Coordinate name suffixes require by this indexer."""

    @property
    @classmethod
    def allow_partial(self) -> bool:
        """Allow some of the suffixes combination missing."""
        return False

    ##

    def _validate_coords(self):
        for coord in self.coords:
            for suffix in self.coord_suffixes:
                coord = f"{coord}_{suffix}"
                if coord in self.dataset.coords:
                    if self.allow_partial:
                        break
                else:
                    if not self.allow_partial:
                        raise ValueError(f'missing coordinate "{coord}"')
            else:
                raise ValueError(
                    f'coordinate "{coord}" does not have any suffix match"'
                )


class GridIndexer(GroupedIndexer):
    coords = ("grid",)
    coord_suffixes = ("x", "y", "z")
    allow_partial = True


class CoordinateIndexer(GroupedIndexer):
    coords = ("coord",)
    coord_suffixes = ("x", "y", "z")
    allow_partial = True
    attrs = ("coord_unit",)


class ViewIndexer(Indexer):
    coords = ("view",)


class TimeIndexer(Indexer):
    coords = ("time",)


class ChannelIndexer(Indexer):
    coords = ("channel",)
