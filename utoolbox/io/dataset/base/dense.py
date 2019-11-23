from .generic import GenericDataset

__all__ = ["ImageDataset", "VolumeDataset"]


class DenseDataset(GenericDataset):
    pass


class ImageDataset(DenseDataset):
    pass


class VolumeDataset(ImageDataset):
    pass
