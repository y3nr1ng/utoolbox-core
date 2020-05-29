import logging
from abc import ABCMeta
from typing import Optional, Tuple

import pandas as pd

from ..generic import BaseDataset, PreloadPriorityOffset
from ..iterators import DatasetIterator

__all__ = ["TiledDataset", "TiledDatasetIterator", "TILE_INDEX_STR"]

logger = logging.getLogger("utoolbox.io.dataset")

# tile position has at most 3-D
# ... unless we figure out how to frak with data in higher dimension
TILE_INDEX_STR = ("tile_x", "tile_y", "tile_z")


class TiledDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        def load_tiling_info():
            self._tile_coords = self._load_mapped_coordinates()

            # build tile shape
            template_names = sorted(list(TILE_INDEX_STR), reverse=True)  # as ZYX order
            index, shape = {}, []
            try:
                for i, name in enumerate(template_names):
                    if name not in self.tile_coords.index.names:
                        if i > 0:
                            # we only add dangling dimension for the slowest axis
                            shape.append(1)
                        continue
                    unique = self.tile_coords.index.get_level_values(name).unique()
                    index[name] = unique
                    shape.append(len(unique))
            except AttributeError:
                # trigger by `self.tile_coords.index`, which should be None
                self._tile_shape = (1,)
            else:
                self._tile_shape = tuple(shape)

            # build index
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
    def tile_index_str(self) -> Optional[Tuple[str]]:
        """Index string used in this dataset."""
        try:
            index_str = list(self.tile_coords.index.names)
        except AttributeError:
            index_str = None
        else:
            index_str.sort(reverse=True)  # as ZYX order
            index_str = tuple(index_str)
        return index_str

    @property
    def tile_shape(self):
        return self._tile_shape

    ##

    def flip_tiling_axes(self, axes):
        axes = [f"tile_{ax}" for ax in axes]

        # flip inventory
        for axis in axes:
            # lookup multiindex numerical index
            i = self.inventory.index.names.index(axis)
            # original values
            values = self.inventory.index.levels[i]
            # flip
            values = values.max() - values
            self.inventory.index.set_levels(values, level=axis, inplace=True)
        self.inventory.sort_index(inplace=True)

        # flip coordinate list (index)
        for axis in axes:
            # lookup multiindex numerical index
            i = self.inventory.index.names.index(axis)
            # original values
            values = self.inventory.index.levels[i]
            # flip
            values = values.max() - values
            self.tile_coords.index.set_levels(values, level=axis, inplace=True)
        # flip coordinate list (coords)
        for axis in axes:
            axis = axis.split("_")[1]
            axis = f"{axis}_coord"
            self.tile_coords[axis] *= -1

    def remap_tiling_axes(self, mapping):
        mapping = {f"tile_{src}": f"tile_{dst}" for src, dst in mapping.items()}

        # rename inventory
        self.inventory.index.rename(
            mapping.values(), level=mapping.keys(), inplace=True
        )

        # rename coordinate list (index)
        self.tile_coords.index.rename(
            mapping.values(), level=mapping.keys(), inplace=True
        )
        # rename coordinate list (value)
        mapping = {f"{src}_coord": f"{dst}_coord" for src, dst in mapping.items()}
        self.tile_coords.rename(mapping, axis="columns", inplace=True)

    ##

    def _load_coordinates(self) -> pd.DataFrame:
        """
        Load the raw coordinates recorded during acquisition. Using TILE_INDEX_STR as 
        header.

        Returns:
            (pd.DataFrame): result coordinates as a DataFrame
        """
        raise NotImplementedError

    def _load_index(self) -> pd.DataFrame:
        """
        Discrete tile index. Using TILE_INDEX_STR as header.

        Returns:
            (pd.DataFrame): these are used as the index
        """
        raise NotImplementedError

    def _infer_index_from_coords(self, coords: pd.DataFrame) -> pd.DataFrame:
        """
        Discrete tile index. Using TILE_INDEX_STR as header.

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
        # remap back
        coord_names_mapping = {v: k for k, v in coord_names_mapping.items()}

        return df


class TiledDatasetIterator(DatasetIterator):
    """
    Iterator for tiled dataset.

    Args:
        dataset (TiledDataset): source dataset
        axes (str, optional): order of axes to loop over
        return_format (str, optional): return as 'index', 'coord' or 'both'
        **kwargs: additional keyword arguments
    """

    def __init__(
        self, dataset: TiledDataset, *, axes="zyx", return_format="index", **kwargs
    ):
        self._return_format = return_format

        # restore axis name
        axes = [f"tile_{axis}" for axis in axes]
        if any(axis not in TILE_INDEX_STR for axis in axes):
            # FIXME use tile_index_str
            desc = ", ".join(f'"{axis}"' for axis in TILE_INDEX_STR)
            raise ValueError(f"axis can only contain {desc}")

        # drop unsupported axis
        index = [axis for axis in axes if axis in dataset.index.names]
        index_diff = set(index) ^ set(axes)
        if index_diff:
            desc = ", ".join(f'"{axis}"' for axis in index_diff)
            logger.debug(f"found unused index, dropping {desc}")

        super().__init__(dataset, index=index, **kwargs)

    def __iter__(self):
        requires_lookup = self.return_key and (self.return_format != "index")

        if requires_lookup:
            # build lookup table for *_coord columns
            key_header = []
            for axis in self.index:
                axis = axis.split("_")[1]
                axis = f"{axis}_coord"
                key_header.append(axis)

        for result in super().__iter__():
            if requires_lookup:
                # return: COORD
                key_index, selected = result

                if key_index:
                    # select the row by matching the multi-index using inner join
                    key_coord = pd.merge(
                        self.dataset.tile_coords,
                        selected.rename("uuid"),
                        how="inner",
                        left_index=True,
                        right_index=True,
                    )
                    # build key order from previous selection order
                    key_coord = tuple(
                        key_coord[header].iloc[0] for header in key_header
                    )
                else:
                    key_coord = None  # not a tiled dataset, nothing to lookup

                if self.return_format == "both":
                    key = key_index, key_coord
                else:
                    # we are already translating the keys, we either have to return
                    # both keys or only the translated one
                    key = key_coord

                yield key, selected
            else:
                # return: NO KEY / INDEX
                yield result

    ##

    @property
    def return_format(self) -> str:
        return self._return_format


class TiledSlabDatasetIterator(TiledDatasetIterator):
    """
    Treat Z slices as slabs. 

    This iterator batches tiles within the same layers or multiple layers (a slab).
    """

    # TODO
