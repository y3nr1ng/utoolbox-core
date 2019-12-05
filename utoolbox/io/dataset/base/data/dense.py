from abc import ABCMeta, abstractmethod
import logging
from itertools import product

import xarray as xr

from ..generic import GenericDataset

__all__ = ["DenseDataset"]

logger = logging.getLogger(__name__)


class DenseDataset(GenericDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    ##

    @property
    @abstractmethod
    def read_func(self):
        """
        callable(URI, SHAPE, DTYPE)
        """
        pass

    ##

    def preload(self):
        shape, dtype = self._load_array_info()

        data_vars = self.dataset.data_vars.keys()
        coords = self.dataset.coords
        coord_keys, coord_values = (
            list(coords.keys()),
            list(v.values for v in coords.values()),
        )

        logger.debug("preloading...")
        for data_var in data_vars:
            logger.debug(f"> data_var: {data_var}")
            for coord in product(*coord_values):
                coord = {k: v for k, v in zip(coord_keys, coord)}
                logger.debug(f">> coord: {coord}")
                file_list = self._retrieve_file_list(data_var, coord)
                if file_list:
                    array = self.read_func(file_list, shape, dtype)
                    array = array.assign_coords(coord)
                    self.dataset[data_var] = array
                else:
                    logger.warning(
                        f'missing data, DATA_VAR "{data_var}", COORD "{coord}"'
                    )
                

    ##

    @abstractmethod
    def _load_array_info(self):
        """Load single data var info, in order to rebuild it in Dask."""
        pass

    @abstractmethod
    def _retrieve_file_list(self, data_var, coords):
        pass
