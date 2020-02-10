import logging

from utoolbox.io.dataset import (
    MicroManagerV1Dataset,
    MicroManagerV2Dataset,
    LatticeScopeDataset,
    LatticeScopeTiledDataset,
    SmartSpimDataset,
)
from .base import UnsupportedDatasetError

__all__ = ["open_dataset"]

logger = logging.getLogger(__name__)


def open_dataset(path):
    klass = [
        # uManager
        MicroManagerV2Dataset,
        MicroManagerV1Dataset,
        # LatticeScope
        LatticeScopeTiledDataset,
        LatticeScopeDataset,
        # SmartSPIM
        SmartSpimDataset,
    ]
    for _klass in klass:
        try:
            ds = _klass(path)
            logger.debug(f'"{path}" is a "{_klass}"')
            break
        except UnsupportedDatasetError:
            logger.debug(f'not a "{_klass}"')
    else:
        raise UnsupportedDatasetError("no supported dataset format")
    return ds
