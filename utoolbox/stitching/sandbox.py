from collections import namedtuple
import logging
import uuid

__all__ = [
    'Sandbox'
]

logger = logging.getLogger(__name__)

BoundingBox = namedtuple(
    'BoundingBox', 
    ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
)

class Brick(object):
    def __init__(self, data, bbox, parent_id=None):
        """
        Parameters
        ----------
        data : np.ndarray
            Represented data.
        bbox : BoundingBox
            Bounding box dimension in absolute coordinate.
        parent : int
            Integer ID of the parent.
        """
        assert data.ndim == 3
        if len(bbox)/2 != data.ndim:
            raise ValueError("bounding box dimension mismatch")
        self._data = data
        self._bbox = BoundingBox(*bbox)

        self.parent_id = parent_id
        self._id = uuid.uuid4().int>>64

    @property
    def bbox(self):
        return self._bbox

    @property
    def data(self):
        return self._data

    @property
    def id(self):
        """A 64-bit random ID from UUID."""
        return self._id


class Sandbox(object):
    def __init__(self):
        pass
    
    def add(self, brick):
        pass
    
    def remove(self, brick):
        pass
    
    def consolidate(self, id=None):
        """
        Consolidate a specific brick in the sandbox, otherwise, all the bricks 
        are consolidated. 

        Parameter
        ---------
        id : int
            ID of the brick to be consolidate, `None` to consolidate ALL.
        
        Note
        ----
        After consolidation occurs, no matter it is global or not, dimension of 
        the sandbox will be fixed.
        """
        pass
