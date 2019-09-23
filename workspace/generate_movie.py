from math import floor
import os

import av
import imageio

from utoolbox.data.datastore import FolderDatastore

root = 'Movie_decon'
ds = FolderDatastore(root, read_func=imageio.imread, extensions=['tif'])
framerate = 24

# dummy read
ny, nx = next(iter(ds.values())).shape[:2]
ny, nx = floor(ny/2)*2, floor(nx/2)*2

# expand path
root = os.path.abspath(root)
parent, basename = os.path.dirname(root), os.path.basename(root)
out_path = os.path.join(parent, "{}.mp4".format(basename))

# create new container
out = av.open(out_path, 'w')
stream = out.add_stream('h264', str(framerate))
stream.width = nx
stream.height = ny
stream.bit_rate = 8e6

for key, im in ds.items():
    print(key)

    frame = av.VideoFrame.from_ndarray(im[:ny, :nx], format='rgb24')
    packet = stream.encode(frame)
    out.mux(packet)

# flush
packet = stream.encode(None)
out.mux(packet)
out.close()