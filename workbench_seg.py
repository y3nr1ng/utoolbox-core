import logging

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = logging.getLogger(__name__)

from utoolbox.container import Volume

mem_data = Volume("data/20171201_RFiSHp2aLFCYC/decon/488/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif")
print(mem_data.shape)
print(mem_data.dtype)

from skimage.exposure import rescale_intensity
mem_data = rescale_intensity(mem_data, out_range=(0., 1.))

import numpy as np
seg_shape = np.ceil(np.asarray(mem_data.shape)/5)
n_segments = np.prod(seg_shape, dtype=np.int32)
print("n_segments={}".format(n_segments))

from utoolbox.segmentation import slic
segments = slic(mem_data, n_segments=n_segments)
print(segments.dtype)

import imageio
imageio.volwrite("data/segments.tif", segments.astype(np.int32))

import imageio
labels = imageio.volread("data/segments.tif")

import numpy as np
i_max = np.amax(labels)
i_min = np.amin(labels)
print("label range [{}, {})".format(i_min, i_max+1))
lut = np.arange(i_min, i_max+1, dtype=np.int32)

import random
random.shuffle(lut)
labels_shuffle = lut[labels]

imageio.volwrite("data/segments_shuffle.tif", labels_shuffle)

from utoolbox.container import Volume
mem_data = Volume("data/20171201_RFiSHp2aLFCYC/decon/488/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif")

histogram = np.zeros_like(lut, dtype=np.float32)
n_histogram = np.zeros_like(lut, dtype=np.int32)

result = np.zeros_like(labels)

mem_data = list(mem_data.ravel())
labels = list(labels.ravel())
for M, L in zip(mem_data, labels):
    histogram[L] += M
    n_histogram[L] += 1
histogram = np.divide(histogram, n_histogram)

np.save("data/histogram.npy", histogram)

import numpy as np
histogram = np.load("data/histogram.npy")

import imageio
labels = imageio.volread("data/segments.tif")

dilated = histogram[labels]
imageio.volwrite("data/dilated.tif", dilated.astype(np.int32))

from skimage.filters import scharr
edges = np.zeros_like(dilated)
for z, s in enumerate(dilated):
    edges[z, ...] = scharr(s)
imageio.volwrite("data/edges.tif", edges.astype(np.int32))

raise RuntimeError

data = np.sort(histogram)
ranking = np.cumsum(data) / np.sum(data)

cutoff = 0.05
i_lo = np.searchsorted(ranking, cutoff)
i_hi = np.searchsorted(ranking, 1.-cutoff)
th_lo = data[i_lo]
th_hi = data[i_hi]

print(th_lo)
print(th_hi)

mask = np.zeros_like(dilated)
mask_lo = (dilated < th_lo) & (mask == 0)
mask_hi = (dilated > th_hi) & (mask == 0)
mask[mask_lo] = 1
mask[mask_hi] = 2

print("remains {} elements".format(np.count_nonzero(mask == 0)))
