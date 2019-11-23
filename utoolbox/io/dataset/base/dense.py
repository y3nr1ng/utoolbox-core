from .generic import GenericDataset

__all__ = ["ImageDataset", "VolumeDataset"]


class DenseDataset(GenericDataset):
    def __init__(self):
        super().__init__()


class ImageDataset(DenseDataset):
    def __init__(self):
        super().__init__()


class VolumeDataset(ImageDataset):
    def __init__(self):
        super().__init__()
