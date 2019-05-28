import logging
import os
from pprint import pprint

import coloredlogs

from utoolbox.container.dataset import SPIMDataset

logging.getLogger("tifffile").setLevel(logging.ERROR)
coloredlogs.install(
    level="DEBUG",
    fmt="%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

dataset = SPIMDataset("~/Documents/Sinica (Data)/Projects/Wen-Chen/20180807/cell5")

pprint(dataset.metadata)

for ch, data in dataset.items():
    print(" << {} >>".format(ch))
    pprint(list(data._uri.items())[:10])

    break