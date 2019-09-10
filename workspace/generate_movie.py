import logging
import os

import coloredlogs
import av
import imageio
import numpy as np

from utoolbox.container.datastore import FileDatastore

logging.getLogger("tifffile").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

##

root = "~/nas/hive_archive_ytliu/20190528_Cornea/G"
framerate = 23.976

##

ds = FileDatastore(
    root, read_func=imageio.imread, extensions=["tif"]
)

# dummy read
ny, nx = next(iter(ds.values())).max(axis=0).shape

# expand path
root = os.path.abspath(root)
parent, basename = os.path.dirname(root), os.path.basename(root)
out_path = os.path.join(parent, "{}.mp4".format(basename))

# create new container
out = av.open(out_path, 'w')
stream = out.add_stream('h264', framerate)
stream.bit_rate = 8e6

for key, im in ds.items():
    logger.info(key)

    frame = av.VideoFrame.from_ndarray(, format=)
    packet = stream.encode(frame)
    out.mux(packet)

# flush
packet = stream.encode(None)
out.mux(packet)
out.close()
