from abc import ABCMeta, abstractmethod

import numpy as np

from ..generic import BaseDataset

__all__ = ["TiledDataset"]


class TiledDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        index, self._tile_coords = self._load_tiling_info()
        assert any(
            key in index.keys() for key in ("tile_x", "tile_y", "tile_z")
        ), "unable to find definition of tiling coordinates"
        self.inventory.update(index)

    ##

    @property
    def tile_coords(self):
        return self._tile_coords

    @property
    def tile_shape(self):
        axes = ("y", "x")
        if "tile_z" in self.tile_coords.columns:
            axes = ("z",) + axes
        return tuple(len(self.tile_coords[f"tile_{ax}"].unique()) for ax in axes)

    ##

    def flip_tiling_axes(self, axes):
        axes = [f"tile_{ax}" for ax in axes]

        # flip inventory, multiindex
        for axis in axes:
            # lookup multiindex numerical index
            i = self.inventory.index.names.index(axis)
            # original values
            values = self.inventory.index.levels[i]
            self.inventory.index.set_levels(values * -1, level=axis, inplace=True)
        self.inventory.sort_index(inplace=True)

        # flip coordinates, dataframe
        for axis in axes:
            self.tile_coords[axis] *= -1

    def remap_tiling_axes(self, mapping):
        # generate complete name
        mapping = {f"tile_{src}": f"tile_{dst}" for src, dst in mapping.items()}

        # rename inventory, multiindex
        self.inventory.index.rename(
            mapping.values(), level=mapping.keys(), inplace=True
        )

        # rename coordinates, dataframe
        self.tile_coords.rename(mapping, axis="columns", inplace=True)

    ##

    @abstractmethod
    def _load_tiling_coordinates(self):
        pass

    def _load_tiling_info(self):
        coords = self._load_tiling_coordinates()
        unique_coords = {ax: coords[ax].unique() for ax in coords}
        return unique_coords, coords
