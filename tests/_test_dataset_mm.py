from pprint import pprint

import coloredlogs
import numpy as np

from utoolbox.data.dataset import MicroManagerDataset

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s  %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

ds = MicroManagerDataset("perfuse_lectin594_poststain_lectin647_5")

for name, im in ds["640_10X"].items():
    pprint(im)
