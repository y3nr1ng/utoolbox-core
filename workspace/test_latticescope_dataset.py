import coloredlogs

from utoolbox.io.dataset import LatticeScopeTiledDataset

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

ds = LatticeScopeTiledDataset("X:/ARod/20191212_4F/flybrain_1")
print(ds.inventory)
