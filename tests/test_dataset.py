# pylint: disable=W0612

import pytest
from pytest import fixture

import utoolbox.latticescope as llsm
from utoolbox.latticescope import sort_by_timestamp

@fixture
def path():
    return "mock_dataset"

@fixture
def ds(path):
    return llsm.Dataset(path, refactor=False)

def test_invalid_root():
    with pytest.raises(FileNotFoundError) as _:
        ds = llsm.Dataset('INVALID')

def test_correct_read(path):
    ds = llsm.Dataset(path, refactor=False)

def test_correct_partitions(ds):
    pass


if __name__ == '__main__':
    from pprint import pprint

    ##### LOAD FILE #####
    path = "mock_dataset"
    ds = llsm.Dataset(path, refactor=False)


    ##### DUMP INVENTORY #####
    pprint(ds.settings)

    pprint(ds.datastore)
    for k, v in ds.datastore.items():
        print(" << {} >>".format(k))
        pprint(v.files)


    ##### SORT #####
    sort_by_timestamp(ds)
    pprint(ds.datastore)
    for k, v in ds.datastore.items():
        print(" << {} >>".format(k))
        pprint(v.files)