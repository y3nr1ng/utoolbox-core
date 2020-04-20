import os

from utoolbox.io.dataset import LatticeScopeTiledDataset, ZarrDataset


def main():
    pwd = os.path.abspath(__file__)
    cwd = os.path.dirname(pwd)
    parent = os.path.dirname(cwd)

    ds_dir = os.path.join(parent, "data", "demo_3D_2x2x2_CMTKG-V3")
    print(ds_dir)

    raise RuntimeError("DEBUG")
    ds = LatticeScopeTiledDataset.load(ds_dir)


if __name__ == "__main__":
    main()
