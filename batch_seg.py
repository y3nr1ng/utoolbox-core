import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import imageio

from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries

from scipy.ndimage.measurements import center_of_mass

from utoolbox.container import Volume
from utoolbox.analysis.roi import extract_mask, mask_to_contour


source_folder = "data/RFiSHp2aLFCYC/decon/488"

def load_data(root):
    files = []
    for fname in os.listdir(root):
        #TODO more sophisticate filename filter
        if fname.endswith(".tif"):
            files.append(os.path.join(root, fname))
    files = files[0:3]  #DEBUG
    print("{} data founded under \"{}\"".format(len(files), root))

    #TODO sort by timestamp

    for index, fname in enumerate(files):
        print("[{}] {}".format(index, os.path.basename(fname)))
        data = Volume(fname)
        data = rescale_intensity(data, out_range=(0., 1.))

        # XY MIP
        data = np.amax(data, axis=0)

        yield data

masks = extract_mask(load_data(source_folder))
for i, m in enumerate(masks):
    m = rescale_intensity(m.astype(np.int32))
    imageio.imwrite("data/ruffling_2d/mask_{:03}.tif".format(i), m)

contours = mask_to_contour(masks)
for i, c in enumerate(contours):
    c = rescale_intensity(c.astype(np.int32))
    imageio.imwrite("data/ruffling_2d/contour_{:03}.tif".format(i), c)

raise RuntimeError

indices = np.where(contour == True)
coords = pd.DataFrame({'x': indices[1], 'y': indices[0]})


# Find COM.
com = center_of_mass(mask)
print("({:.4f}, {:.4f})".format(com[0], com[1]))


# Convert contour to polar coordinate system (with respect to the COM).
r = np.sqrt((indices[1]-com[0])**2 + (indices[0]-com[1])**2)
theta = np.arctan2((indices[0]-com[1]), (indices[1]-com[0]))

coords['r'] = pd.Series(r, index=coords.index)
coords['theta'] = pd.Series(theta, index=coords.index)


# Bin to specified regions for polar chart visualization and compute averaged directional distance.
n_bins = 24
bins = np.linspace(-np.pi, np.pi, n_bins+1, dtype=np.float32)
cardinal = np.digitize(theta, bins)-1
coords['cardinal'] = pd.Series(cardinal, index=coords.index)

dist = np.zeros(n_bins, dtype=np.float32)
n_dist = np.zeros(n_bins, dtype=np.int32)
for c, d in zip(cardinal, r):
    dist[c] = max(dist[c], d)
dist /= np.amax(dist)

data_contour = mark_boundaries(data, contour, color=(1, 1, 1))
plt.subplot(1, 2, 1)
plt.imshow(data_contour, cmap='jet')
plt.scatter(com[1], com[0], s=250, c='red', marker='+')

pol_grid = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
ax = plt.subplot(1, 2, 2, projection='polar')
ax.set_theta_direction(-1)
ax.set_theta_zero_location('E')
ax.bar(pol_grid, dist, width=2*np.pi/n_bins, color='lightblue', edgecolor='k')
