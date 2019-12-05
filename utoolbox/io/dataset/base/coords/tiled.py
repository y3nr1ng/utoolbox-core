from abc import ABCMeta, abstractmethod

import numpy as np

from ..generic import GenericDataset

__all__ = ["TiledDataset"]


class TiledDataset(GenericDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        index, self._tile_coords = self._load_tiling_info()
        assert any(
            key in index.keys() for key in ("tile_x", "tile_y", "tile_z")
        ), "unable to find definition of tiling coordinates"
        self._dataset = self.dataset.assign_coords(index)

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
        unique_coords = {
            # NOTE np.unique returns sorted unique values
            k: np.unique(v)
            for k, v in coords.items()
        }
        return unique_coords, coords
