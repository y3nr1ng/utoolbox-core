from pprint import pprint

import coloredlogs
import numpy as np

from utoolbox.data.dataset import MicroManagerDataset

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s  %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

dataset = MicroManagerDataset("Z:/charm/20181009_ExM_4x_hippocampus", merge=True)
with dataset['488'] as source:
    print(source[110])

