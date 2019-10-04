import logging
import os
from pprint import pprint

import coloredlogs

from utoolbox.container.dataset import SPIMDataset

logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

dataset = SPIMDataset("~/Documents/Sinica (Data)/Projects/Wen-Chen/20180807/cell5")

pprint(dataset.metadata)

for ch, data in dataset.items():
    print(" << {} >>".format(ch))
    for i, array in enumerate(list(data._uri.values())[:5]):
        print("[{}], {}, {}".format(i, array.shape, array.dtype))
