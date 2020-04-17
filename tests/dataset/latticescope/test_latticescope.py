import pytest

from utoolbox.io import dataset

# TODO switch between tiled and non-tiled
@pytest.fixture(
    cls=[dataset.LatticeScopeTiledDataset], path=["../data/demo_3x3x1_CMTKG-V3"]
)
def dataset(cls, path):
    return cls.load(path)


def test_open_dataset(dataset):
    assert dataset._can_read() == True
