from pprint import pprint
from utoolbox.io.dataset import MicroManagerV2Dataset


def test_metadata_v2():
    class DummyClass:
        root_dir = ".."

    ds = DummyClass()
    ds.metadata = MicroManagerV2Dataset._load_metadata(ds)

    coords, index, labels = MicroManagerV2Dataset._parse_position_list(ds)

    pprint(coords)
    pprint(index)
    pprint(labels)


if __name__ == "__main__":
    test_metadata_v2()
