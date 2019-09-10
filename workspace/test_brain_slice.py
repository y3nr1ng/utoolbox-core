from math import ceil

import cupy as cp
from cupyx.scipy.ndimage import rotate, zoom
import imageio 
import numpy as np

from utoolbox.util.decorator import timeit

@timeit
def func(im_in, out_res=(2160, 3840), p0=(876, 144), shape=(4096, 5592), bits=16, factor=0.5):
    im_in = cp.asarray(im_in)

    # histogram
    hist, edges = cp.histogram(im_in, bins=256)
    edges = ((edges[:-1] + edges[1:])/2).astype(im_in.dtype)
    hist = cp.asnumpy(hist)

    nelem = im_in.size
    limit = nelem / 10
    autothreshold= 5000
    threshold = nelem / autothreshold

    hmin = -1
    for i, cnt in enumerate(hist):
        if cnt > limit:
            continue
        if cnt > threshold:
            hmin = i
            break

    hmax = -1
    for i, cnt in reversed(list(enumerate(hist))):
        if cnt > limit:
            continue
        if cnt > threshold:
            hmax = i
            break

    vmin, vmax = edges[hmin], edges[hmax]
    print(f'min={vmin}, max={vmax}')

    lookup = cp.ElementwiseKernel(
        'T in', 'T out', 
        f'''
        float fin = ((float)in - {vmin}) / ({vmax}-{vmin});
        if (fin < 0) {{
            fin = 0;
        }} else if (fin > 1) {{
            fin = 1;
        }}
        out = (T)(fin * 65535);
        ''',
        'lookup'
    )
    im_hist = lookup(im_in)

    # rotate to horizontal view
    im_rot = rotate(im_hist, 103)

    # crop and zoom
    im_crop = im_rot[p0[0]:p0[0]+shape[0], p0[1]:p0[1]+shape[1]]
    ratio = [sout/sin for sin, sout in zip(im_crop.shape, out_res)]
    im_zoom = zoom(im_crop, max(ratio))

    # gamma correction
    #bases = cp.float32(bits**16)
    #im_gamma = bases * ((im_in/bases) ** factor)

    im_out = cp.asnumpy(im_zoom).astype(np.uint16)

    return im_out

path = 'E:\\Nature COPY\\Chia-Ming\\BigSheet_DS\\TIF_ch1s.tif'

im_in = imageio.volread(path)
im_out = None
for i, im in enumerate(im_in):
    im = func(im)
    try:
        im_out[i, ...] = im
    except TypeError:
        im_out = np.empty((im_in.shape[0], ) + im.shape)
        im_out[i, ...] = im
imwrite('output.tif', im_out)
