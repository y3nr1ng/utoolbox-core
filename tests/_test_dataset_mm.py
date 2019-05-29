import os
from pprint import pprint

import coloredlogs

from utoolbox.container.dataset import MicroManagerDataset

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

test_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(test_dir, "mm_dataset")

dataset = MicroManagerDataset(path)
pprint(dataset.metadata)
