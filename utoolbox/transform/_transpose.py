import numpy as np

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

__all__ = [
    'Transpose3'
]

class Transpose3(metaclass=AbstractAlgorithm):
    """
    Return contiguous transposed 3D array.
    """
    def __call__(self, A, axis):
        if 'x' not in axis:
            At = self._trans_yz(A)
        elif 'y' not in axis:
            At = self._trans_xz(A)
        elif 'z' not in axis:
            At = self._trans_xy(A)
        else:
            raise ValueError("invalid transpose plane")
        return np.ascontiguousarray(At)

    @interface
    def _trans_xy(self, A):
        pass
    
    @interface
    def _trans_yz(self, A):
        pass
    
    @interface
    def _trans_xz(self, A):
        pass

class Transpose3_CPU(Transpose3):
    _strategy = ImplTypes.CPU_ONLY

    def _trans_xy(self, A):
        return np.transpose(A, (2, 0, 1))
    
    def _trans_yz(self, A):
        return np.transpose(A, (1, 2, 0))
    
    def _trans_xz(self, A):
        return np.transpose(A, (0, 1, 2))
