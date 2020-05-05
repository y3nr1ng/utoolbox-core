import logging
from abc import ABCMeta, abstractmethod
from typing import List, Mapping, Tuple, Union

import numpy as np
import pandas as pd

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["TiledDataset", "TiledDatasetIterator", "TILED_INDEX"]

logger = logging.getLogger("utoolbox.io.dataset")

# tile position has at most 3-D
# ... unless we figure out how to frak with data in higher dimension
TILED_INDEX = ("tile_x", "tile_y", "tile_z")


class TiledDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        def load_tiling_info():
            self._tile_coords = self._load_mapped_coordinates()

            # build tile shape
            axes = ("y", "x")
            if "tile_z" in self._tile_coords.index.names:
                axes = ("z",) + axes
            index, shape = {}, []
            for ax in axes:
                name = f"tile_{ax}"
                unique = self._tile_coords.index.get_level_values(name).unique()
                index[name] = unique
                shape.append(len(unique))
            self._tile_shape = tuple(shape)

            # build index
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
        """Coordinate lookup table."""
        return self._tile_coords

    @property
    def tile_index(self):
        return self._tile_index

    @property
    def tile_shape(self):
        return self._tile_shape

    ##

    def flip_tiling_axes(self, axes):
        # TODO
        axes = [f"tile_{ax}" for ax in axes]

        # flip inventory, multi-index
        for axis in axes:
            # lookup multiindex numerical index
            i = self.inventory.index.names.index(axis)
            # original values
            values = self.inventory.index.levels[i]
            # flip
            values -= max(values)
            values *= -1
            self.inventory.index.set_levels(values, level=axis, inplace=True)
        self.inventory.sort_index(inplace=True)

        # flip internal list, dataframe
        for axis in axes:
            # index
            self.tile_index[axis] -= self.tile_index[axis].max()
            self.tile_index[axis] *= -1
            # coords
            self.tile_coords[axis] *= -1

    def remap_tiling_axes(self, mapping):
        # generate complete name
        mapping = {f"tile_{src}": f"tile_{dst}" for src, dst in mapping.items()}

        # rename inventory, multi-index
        self.inventory.index.rename(
            mapping.values(), level=mapping.keys(), inplace=True
        )

        # rename internal list, dataframe
        self.tile_index.rename(mapping, axis="columns", inplace=True)
        self.tile_coords.rename(mapping, axis="columns", inplace=True)

    ##

    def _load_coordinates(self) -> pd.DataFrame:
        """
        Load the raw coordinates recorded during acquisition. Using TILED_INDEX as 
        header.

        Returns:
            (pd.DataFrame): result coordinates as a DataFrame
        """
        raise NotImplementedError

    def _load_index(self) -> pd.DataFrame:
        """
        Discrete tile index. Using TILED_INDEX as header.

        Returns:
            (pd.DataFrame): these are used as the index
        """
        raise NotImplementedError

    def _infer_index_from_coords(self, coords: pd.DataFrame) -> pd.DataFrame:
        """
        Discrete tile index. Using TILED_INDEX as header.

        Returns:
            (pd.DataFrame): these are used as the index
        """
        logger.warning("infer index by raw coordinates may lead to unwanted error")

        index = coords.rank(axis="index", method="dense", ascending=True)
        # integer 0-based index
        index = index.astype(int) - 1
        return index

    def _load_mapped_coordinates(self) -> pd.DataFrame:
        try:
            coords = self._load_coordinates()
        except NotImplementedError:
            raise NotImplementedError(
                "must implement either `_load_coordinates` or  `_load_mapped_coordinates`"
            )
        try:
            index = self._load_index()
        except NotImplementedError:
            index = self._infer_index_from_coords(coords)

        # DataFrame has to be the same size
        if len(coords) != len(index):
            raise ValueError("cannot map tile index table to coordinates")

        # rename coords
        coord_names_mapping = {}
        for old_name in coords.columns:
            ax = old_name.split("_")[1]
            new_name = f"{ax}_coord"
            coord_names_mapping[old_name] = new_name
        coords.rename(coord_names_mapping, axis="columns", inplace=True)

        # build multi-index
        df = pd.concat([index, coords], axis="columns")
        df.set_index(index.columns.to_list(), inplace=True)

        return df


class TiledDatasetIterator(DatasetIterator):
    """
    Iterator for tiled dataset.

    Args:
        dataset (TiledDataset): source dataset
        axis (str, optional): order of tiling axis to tile over with
        return_real_coord (bool, optional): return actual coordinate instad of index
        **kwargs: additional keyword arguments
    """

    def __init__(
        self, dataset: TiledDataset, *, axis="zyx", return_real_coord=False, **kwargs
    ):
        self._return_real_coord = return_real_coord

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
            index = "tile_0"

        super().__init__(dataset, index=index, **kwargs)

    def __iter__(self):
        for result in super().__iter__():
            if self.return_key and self.return_real_coord:
                key, selected = result

                print(">>>")
                print(self.dataset.tile_coords)
                print(self.dataset.tile_index)
                print(selected)
                print("<<<")

                df_sel = selected.index.to_frame(index=False)
                df_sel = df_sel[self.dataset.tile_coords.columns.to_list()]
                print(df_sel)
                # isolate the labels
                print(pd.merge(self.dataset.tile_index, df_sel, how="inner"))
                raise RuntimeError
                yield key, selected
            else:
                yield result

    ##

    @property
    def return_real_coord(self):
        return self._return_real_coord


class TiledSlabDatasetIterator(TiledDatasetIterator):
    """
    Treat Z slices as slabs. 

    This iterator batches tiles within the same layers or multiple layers (a slab).
    """

    # TODO
