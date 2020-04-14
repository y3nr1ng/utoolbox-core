import logging

from ..base import DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset

__all__ = ["ZarrDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class ZarrDataset(DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset):
    pass
