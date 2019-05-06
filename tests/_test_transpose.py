import numpy as np
import pytest
from pytest import fixture

from utoolbox.container import ImplTypes
from utoolbox.transform import Transpose3

@fixture
def tr3_cpu():
    return Transpose3(ImplTypes.CPU_ONLY)
    
def test_tr3_xy_small_cpu(tr3_cpu):
    A = np.random.rand(*[2, 3, 4])
    Ag = np.transpose(A, (2, 0, 1))
    At = tr3_cpu(A, 'xy')
    assert np.array_equal(Ag, At)

def test_tr3_yz_small_cpu(tr3_cpu):
    A = np.random.rand(*[2, 3, 4])
    Ag = np.transpose(A, (1, 2, 0))
    At = tr3_cpu(A, 'yz')
    assert np.array_equal(Ag, At)

def test_tr3_xz_small_cpu(tr3_cpu):
    A = np.random.rand(*[2, 3, 4])
    Ag = np.transpose(A, (0, 1, 2))
    At = tr3_cpu(A, 'xz')
    assert np.array_equal(Ag, At)