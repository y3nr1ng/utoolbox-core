import os
from pprint import pprint

import coloredlogs

from utoolbox.container.dataset import SPIMDataset

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

test_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(test_dir, "../data/20170112_RFiSHp2aLFCYC/raw")

dataset = SPIMDataset(path)
pprint(dataset.metadata.waveform.channels)

for ch, data in dataset.items():
    print(" << {} >>".format(ch))
    pprint(list(data.items()))

"""
##### SORT #####
sort_by_timestamp(ds)
pprint(ds.datastore)
for k, v in ds.datastore.items():
    print(" << {} >>".format(k))
    pprint(v.files)
"""