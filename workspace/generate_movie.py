from math import floor
import os

import imageio

from utoolbox.data import MicroManagerDataset

root = 'Z:/charm/20181009_ExM_4x_hippocampus'
ds = MicroManagerDataset(root)
framerate = 24

for channel, datastore in ds.items():
    with datastore as source:
        for name, data in source.items():
            print(name)
            imageio.imwrite('demo.tif', data)
            break
    break

"""
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
"""