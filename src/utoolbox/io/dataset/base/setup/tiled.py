import logging
from abc import ABCMeta, abstractmethod
from typing import Mapping, Tuple

import numpy as np

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["TiledDataset", "TiledDatasetIterator"]

logger = logging.getLogger("utoolbox.io.dataset")

TILED_INDEX = ("tile_x", "tile_y", "tile_z")


class TiledDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        def load_tiling_info():
            index, self._tile_coords = self._load_tiling_info()
            assert any(
                key in index.keys() for key in TILED_INDEX
            ), "unknown tiling definition"
            self._update_inventory_index(index)

        self.register_preload_func(
            load_tiling_info, priority=PreloadPriorityOffset.Metadata
        )

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

    def _load_coordinates(self):
        raise NotImplementedError

    def _load_index(self):
        raise NotImplementedError

    def _load_mapped_coordinates(self) -> Mapping[Tuple[int], Tuple[np.float32]]:
        try:
            index, coords = self._load_index(), self._load_coordinates()
        except NotImplementedError:
            raise NotImplementedError(
                "either `_load_coordinates` or `_load_mapped_coordinates` has to be implemented"
            )
        else:
            if len(index) != len(coords):
                raise ValueError(
                    "index definitions does not match coordinates definitions"
                )
            mapping = {k: v for k, v in zip(index, coords)}

        # TODO use `mapping` keys() to generate inventory index
        
        # TODO any way to simplify this section?
        
        return mapping

    ##

    @abstractmethod
    def _load_tiling_coordinates(self):
        pass

    def _load_tiling_index(self):
        logger.warning("infer index by raw coordinates may lead to unwanted error")
        return None

    def _load_tiling_info(self):
        coords = self._load_tiling_coordinates()
        unique_coords = {ax: coords[ax].unique() for ax in coords}

        from pprint import pprint

        pprint(coords)
        pprint(unique_coords)

        raise RuntimeError("DEBUG, load_tiling_info")

        return unique_coords, coords


class TiledDatasetIterator(DatasetIterator):
    """
    Iterator for tiled dataset.

    Args:
        dataset (TiledDataset): source dataset
        axis (str, optional): order of tiling axis to tile over with
        **kwargs: additional keyword arguments
    """

    def __init__(self, dataset: TiledDataset, *, axis="zyx", **kwargs):
        # restore axis name
        axis = [f"tile_{a}" for a in axis]
        if any(a not in TILED_INDEX for a in axis):
            desc = ", ".join(f'"{a}"' for a in TILED_INDEX)
            raise ValueError(f"axis can only contain {desc}")

        # drop unsupported axis
        index = [a for a in axis if a in dataset.index.names]
        delta = set(index) ^ set(axis)
        if len(delta) > 0:
            desc = ", ".join(f'"{a}"' for a in delta)
            logger.debug(f"found unused index, dropping {desc}")

        if not index:
            # use dummy axis header to trigger KeyError
            index = "tile_"

        super().__init__(dataset, index=index, **kwargs)


class TiledSlabDatasetIterator(TiledDatasetIterator):
    """
    Treat Z slices as slabs. 

    This iterator batches tiles within the same layers or multiple layers (a slab).
    """

    # TODO
