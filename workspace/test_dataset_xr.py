# flake8:noqa
from utoolbox.io.dataset_xr import formats

print(formats)

from utoolbox.io.dataset_xr import open_dataset

path = "utoolbox-core/workspace/data/20200704_kidney_demo-2_CamA"
with open_dataset(path, "rw") as dataset:
    pass

raise RuntimeError("DEBUG")

