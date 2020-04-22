from typing import Tuple
from abc import ABCMeta, abstractmethod

__all__ = ["DimensionalDataset"]


class DimensionalDataset(metaclass=ABCMeta):
    """
    A dataset with predefined dimensional requirements, a _dimension_ can be a 
    temporal axis, different color, spatial location.. etc.
    """

    # Please define class attribute `index` in the subclasses. This is enforced
    # using abstract classmethod below.
    # index = tuple()

    ##

    @property
    @classmethod
    @abstractmethod
    def index(cls) -> Tuple[str]:
        raise NotImplementedError("dimensional index has to ")
