import os

import pytest

from utoolbox.io.dataset import LatticeScopeTiledDataset
from utoolbox.io.dataset.base import UnsupportedDatasetError


@pytest.fixture
def path():
    cwd = os.path.dirname(__file__)
    return os.path.join(cwd, "data", "demo_3x3x1_CMTKG-V3")


@pytest.fixture
def dataset(path):
    """Manually create the dataset object, not loaded yet."""
    dataset = LatticeScopeTiledDataset(path)
    return dataset


def test_load(path):
    try:
        LatticeScopeTiledDataset.load(path)
    except UnsupportedDatasetError:
        pytest.fail("unexpected UnsupportedDatasetError")
