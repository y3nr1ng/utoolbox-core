import logging
import os

import coloredlogs
import pandas as pd
from tqdm import tqdm

from utoolbox.smlm.utils import roll_by_frame

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)


##### LOAD DATA #####
df = pd.read_csv('part1_corrected.csv')
wnd_size = 300


##### START ROLLING #####
func, n_frames = roll_by_frame(df, wnd_size=wnd_size)
for index, frame in enumerate(tqdm(func, total=n_frames)):
    frame.to_csv("{:04d}.csv".format(index), index=False)
