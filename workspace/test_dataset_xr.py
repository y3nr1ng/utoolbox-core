from utoolbox.io.dataset_xr import formats
print(formats)

from utoolbox.io.dataset_xr import open_dataset

path = "utoolbox-core/workspace/data/20200704_kidney_demo-2_CamA"
dataset = open_dataset(path, "r")

raise RuntimeError('DEBUG')