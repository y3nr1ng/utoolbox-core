import logging
import os

import av
import coloredlogs
import cupy as cp
import imageio
import numpy as np

from utoolbox.data.datastore import FolderDatastore
from utoolbox.data.io.amira import AmiraColormap
from utoolbox.exposure import auto_contrast

logging.getLogger("tifffile").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

##

root = "Z:/ytliu/09112019_c1"
framerate = 23.976

##

ds = FolderDatastore(root, read_func=imageio.volread, extensions=["tif"])

# dummy read
ny, nx = next(iter(ds.values())).max(axis=0).shape

# expand path
root = os.path.abspath(root)
parent, basename = os.path.dirname(root), os.path.basename(root)
out_path = os.path.join(parent, "{}.mp4".format(basename))

# load colormap
cm = AmiraColormap("volrenGlow.am")
cm = cm.data
# drop alpha channel
cm = cm[:, :3]
# scale to 8-bit
cm = (255 * cm).astype(np.uint8)

# create new container
out = av.open(out_path, "w")
stream = out.add_stream("h264", str(framerate))
stream.bit_rate = 8e6

# loop through frames
for key, im in ds.items():
    logger.info(key)

    im = cp.asarray(im)

    im = auto_contrast(im, auto_threshold=1000000)
    im *= 255

    im = im.max(axis=0)

    im = im.astype(cp.uint8)
    im = cp.asnumpy(im)

    frame = cm[im]
    
    frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
    packet = stream.encode(frame)
    out.mux(packet)

# flush
packet = stream.encode(None)
out.mux(packet)
out.close()
