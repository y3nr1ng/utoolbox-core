import logging
from pprint import pprint

import coloredlogs

from utoolbox.data import SPIMDataset

logging.getLogger("tifffile").setLevel(logging.ERROR)

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

dataset = SPIMDataset("raw")
pprint(dataset.info)

