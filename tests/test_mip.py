import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import fixture

from utoolbox.transform.projections import Orthogonal

@fixture
def data(shape=(64, 128, 256)):
    return np.random.randint(0, 2**16-1, size=shape, dtype=np.uint16)

def test_xy_proj(data):
    cpu = np.max(data, axis=0)
    with Orthogonal(data) as ortho:
        gpu = ortho.xy
    assert_array_equal(cpu, gpu)

def test_xz_proj(data):
    cpu = np.max(data, axis=1)
    with Orthogonal(data) as ortho:
        gpu = ortho.xz
    assert_array_equal(cpu, gpu)

def test_yz_proj(data):
    cpu = np.max(data, axis=2)
    with Orthogonal(data) as ortho:
        gpu = ortho.yz
    assert_array_equal(cpu, gpu)