from abc import ABCMeta, abstractmethod
import os

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, root):
        """
        Parameters
        ----------
        root : str
            Source of the dataset, flat layout.
        """
        if not os.path.exists(root):
            raise FileNotFoundError("invalid dataset source")
        self._root = root
        self._datastore = None
    
    @property
    def datastore(self):
        return self._datastore

    @property
    def root(self):
        return self._root

    @abstractmethod
    def preview(self, view='all'):
        """
        Generate projection view for the dataset.

        Parameters
        ----------
        view : one of ['xy', 'yz', 'xz'], or 'all'
            Projected view to generate, 'all' will composite all views to a 
            single frame.
        """
        return NotImplemented
