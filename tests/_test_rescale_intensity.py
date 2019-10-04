import coloredlogs
import numpy as np
from numpy.testing import assert_array_almost_equal
import pycuda.driver as cuda
import pytest
from pytest import fixture

from utoolbox.container import ImplTypes
from utoolbox.exposure import RescaleIntensity
from utoolbox.parallel.gpu import create_some_context

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

@fixture
def rescale_cpu():
    return RescaleIntensity(ImplTypes.CPU_ONLY)

@fixture
def rescale_gpu():
    ctx = create_some_context()
    ctx.push()

    yield RescaleIntensity(ImplTypes.GPU)

    cuda.Context.pop()

def test_rescale_stretch_cpu(rescale_cpu):
    image = np.array([51, 102, 153], dtype=np.uint8)
    out = rescale_cpu(image)
    assert out.dtype == np.uint8
    assert_array_almost_equal(out, [0, 127, 255])

def test_rescale_shrink_cpu(rescale_cpu):
    image = np.array([51., 102., 153.])
    out = rescale_cpu(image)
    assert_array_almost_equal(out, [0, 0.5, 1])

def test_rescale_in_range_cpu(rescale_cpu):
    image = np.array([51., 102., 153.])
    out = rescale_cpu(image, in_range=(0, 255))
    assert_array_almost_equal(out, [0.2, 0.4, 0.6])

def test_rescale_in_range_clip_cpu(rescale_cpu):
    image = np.array([51., 102., 153.])
    out = rescale_cpu(image, in_range=(0, 102))
    assert_array_almost_equal(out, [0.5, 1, 1])

def test_rescale_out_range_cpu(rescale_cpu):
    image = np.array([-10, 0, 10], dtype=np.int8)
    out = rescale_cpu(image, out_range=(0, 127))
    assert out.dtype == np.int8
    assert_array_almost_equal(out, [0, 63, 127])

def test_rescale_named_in_range_cpu(rescale_cpu):
    """
    array: uint16
    input: uint8 (by name) -> output: uint16
    """
    image = np.array([0, (2**8-1), (2**8-1) + 100], dtype=np.uint16)
    out = rescale_cpu(image, in_range=np.uint8)
    assert_array_almost_equal(out, [0, 2**16-1, 2**16-1])

def test_rescale_named_out_range_cpu(rescale_cpu):
    """
    array: uint16
    input: uint16 -> output: uint8 (by name)
    """
    image = np.array([0, 2**16-1], dtype=np.uint16)
    out = rescale_cpu(image, out_range=np.uint8)
    assert_array_almost_equal(out, [0, 2**8-1])

def test_rescale_stretch_gpu(rescale_gpu):
    image = np.array([51, 102, 153], dtype=np.uint8)
    out = rescale_gpu(image)
    assert out.dtype == np.uint8
    assert_array_almost_equal(out, [0, 127, 255])

def test_rescale_shrink_gpu(rescale_gpu):
    image = np.array([51., 102., 153.])
    out = rescale_gpu(image)
    assert_array_almost_equal(out, [0, 0.5, 1])
