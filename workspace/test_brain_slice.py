from math import ceil
import os

import av
import cupy as cp
from cupyx.scipy.ndimage import rotate, zoom
import imageio
import numpy as np

from utoolbox.exposure import auto_contrast
from utoolbox.utils.decorator import timeit


@timeit
def func(
    im_in, out_res=(2160, 3840), p0=(876, 144), shape=(4096, 5592), bits=16, factor=0.5
):
    im_in = cp.asarray(im_in)

    im_hist = auto_contrast(im_in)

    # rotate to horizontal view
    im_rot = rotate(im_hist, 103)

    # crop and zoom
    im_crop = im_rot[p0[0] : p0[0] + shape[0], p0[1] : p0[1] + shape[1]]
    ratio = [sout / sin for sin, sout in zip(im_crop.shape, out_res)]
    im_zoom = zoom(im_crop, max(ratio))

    # gamma correction
    # bases = cp.float32(bits**16)
    # im_gamma = bases * ((im_in/bases) ** factor)

    # type cast to 8-bit for movies
    im_bit = im_zoom / 65535 * 255

    im_out = cp.asnumpy(im_bit).astype(np.uint8)
    return im_out


@timeit
def main(path, framerate=23.976):
    # expand path
    path = os.path.abspath(path)
    parent, basename = os.path.dirname(path), os.path.basename(path)
    basename, _ = os.path.splitext(basename)
    out_path = os.path.join(parent, "{}.mp4".format(basename))

    # create new container
    out = av.open(out_path, "w")
    stream = out.add_stream("h264", str(framerate))
    stream.bit_rate = 8000000

    im_in = imageio.volread(path)
    im_out = None
    for i, im in enumerate(im_in):
        print(f"z={i}")
        im = func(im)

        frame = av.VideoFrame.from_ndarray(im, format="gray")
        packet = stream.encode(frame)
        out.mux(packet)

    # flush
    packet = stream.encode(None)
    out.mux(packet)
    out.close()


path = [
    "E:\\Nature COPY\\Chia-Ming\\BigSheet_DS\\TIF_ch2s.tif",
    "E:\\Nature COPY\\Chia-Ming\\BigSheet_DS\\TIF_ch3s.tif"
]
for p in path:
    main(p)
