import coloredlogs

from utoolbox.io.dataset import BDVDataset, LatticeScopeTiledDataset

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

src_ds = LatticeScopeTiledDataset("X:/ARod/20191212_4F/flybrain_1")
print(src_ds.inventory)

dst_dir = 'U:/ARod/20191212_4F/flybrain_1_bdv'
BDVDataset.dump(dst_dir, src_ds, pyramid=[(1, 1, 1), (2, 4, 4)])
