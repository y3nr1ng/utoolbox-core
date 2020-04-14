import logging

from utoolbox.io.dataset import (
    MicroManagerV1Dataset,
    MicroManagerV2Dataset,
    LatticeScopeDataset,
    LatticeScopeTiledDataset,
    SmartSpimDataset,
)
from .base import UnsupportedDatasetError

__all__ = ["open_dataset", "SUPPORTED_DATASET_CLASS"]

logger = logging.getLogger("utoolbox.io.dataset")


SUPPORTED_DATASET_CLASS = [
    # uManager
    MicroManagerV2Dataset,
    MicroManagerV1Dataset,
    # LatticeScope
    LatticeScopeTiledDataset,
    LatticeScopeDataset,
    # SmartSPIM
    SmartSpimDataset,
]


def open_dataset(path):
    for _klass in SUPPORTED_DATASET_CLASS:
        try:
            ds = _klass.load(path)
            logger.info(f'"{path}" is a "{_klass}"')
            break
        except UnsupportedDatasetError:
            logger.debug(f'not a "{_klass}"')
    else:
        raise UnsupportedDatasetError("no supported dataset format")
    return ds
