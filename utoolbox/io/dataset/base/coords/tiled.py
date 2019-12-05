from abc import ABCMeta, abstractmethod
from ..generic import GenericDataset

__all__ = ["TiledDataset"]


class TiledDataset(GenericDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        coords = self._load_tiling_positions()
        assert any(
            key in coords.keys() for key in ("tile_x", "tile_y", "tile_z")
        ), "unable to find definition of tiling coordinates"
        self._dataset = self.dataset.assign_coords(coords)

    ##

    ##

    @abstractmethod
    def _load_tiling_positions(self):
        pass
