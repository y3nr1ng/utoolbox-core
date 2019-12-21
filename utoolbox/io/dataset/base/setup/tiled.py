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

    ##

    @abstractmethod
    def _load_tiling_coordinates(self):
        pass

    def _load_tiling_info(self):
        coords = self._load_tiling_coordinates()
        unique_coords = {ax: coords[ax].unique() for ax in coords}
        return unique_coords, coords
