from abc import ABCMeta, abstractmethod
import logging

import pandas as pd

from ..generic import BaseDataset

__all__ = ["DenseDataset"]

logger = logging.getLogger(__name__)


class DenseDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    ##

    def preload(self):
        self._consolidate_inventory()
        assert isinstance(self.inventory, pd.MultiIndex), "inventory not consolidated"

        shape, dtype = self._load_array_info()

        logger.debug("preloading...")
        data_uuid = []
        for coord in self.inventory:
            coord_dict = {k: v for k, v in zip(self.inventory.names, coord)}
            file_list = self._retrieve_file_list(coord_dict)

            if file_list:
                array = self.read_func(file_list, shape, dtype)
                uuid = self._register_data(array)
            else:
                uuid = ""
                coord_str = [f"{k}:{v}" for k, v in coord_dict.items()]
                coord_str = "(" + ", ".join(coord_str) + ")"
                logger.warning(f"missing data, {coord_str}")
            data_uuid.append(uuid)

        # complete the preload process
        self._inventory = pd.Series(data_uuid, index=self.inventory)

    ##

    @abstractmethod
    def _load_array_info(self):
        """Load single data var info, in order to rebuild it in Dask."""
        pass

    @abstractmethod
    def _load_voxel_size(self):
        """Load voxel size."""
        pass

    @abstractmethod
    def _retrieve_file_list(self, coord_dict):
        pass
