import logging

__all__ = ["GenericDataset"]

logger = logging.getLogger(__name__)


class GenericDataset(object):
    def __init__(self):
        pass

    def to_zarr(self):
        try:
            import zarr
        except ImportError:
            logger.error("zarr is not intsalled")

        # TODO collect axes info
