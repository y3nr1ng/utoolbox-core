import logging

from utoolbox.io.dataset import (
    LatticeScopeDataset,
    LatticeScopeTiledDataset,
    MicroManagerV1Dataset,
    MicroManagerV2Dataset,
    SmartSpimDataset,
    ZarrDataset,
)

from .base import UnsupportedDatasetError

__all__ = ["open_dataset", "SUPPORTED_DATASET_CLASS"]

logger = logging.getLogger("utoolbox.io.dataset")


SUPPORTED_DATASET_CLASS = [
    # Zarr
    ZarrDataset,
    # uManager
    MicroManagerV2Dataset,
    MicroManagerV1Dataset,
    # LatticeScope
    LatticeScopeTiledDataset,  # we must test the tiled-form first
    LatticeScopeDataset,
    # SmartSPIM
    SmartSpimDataset,
]


def open_dataset(path, show_trace=False):
    for _klass in SUPPORTED_DATASET_CLASS:
        try:
            ds = _klass.load(path)
            logger.info(f'"{path}" is a "{_klass.__name__}"')
            break
        except UnsupportedDatasetError:
            print_func = logger.exception if show_trace else logger.debug
            print_func(f'not a "{_klass.__name__}"')
    else:
        raise UnsupportedDatasetError("no supported dataset format")
    return ds
