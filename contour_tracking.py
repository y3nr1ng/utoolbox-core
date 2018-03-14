import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.measurements import center_of_mass

points = pd.read_pickle("data/ruffling_2d/contours.pkl")

n_bins = 24
bins = np.linspace(-np.pi, np.pi, n_bins+1, dtype=np.float)

cxs = []
cys = []
cardinal_map = {}
for iframe in points.index.levels[0]:
    x = points.loc[iframe, 'x'].values.astype(np.float)
    y = points.loc[iframe, 'y'].values.astype(np.float)

    cx, cy = np.mean(x), np.mean(y)
    cxs.append(cx)
    cys.append(cy)

    # convert to polar coordinate system with respect to the COM
    x -= cx
    y -= cy
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # cardinal direction
    cardinal = np.digitize(theta, bins) - 1

    # directional distance, maximum
    distance = np.zeros(n_bins, dtype=np.float)
    for c, r in zip(cardinal, radius):
        distance[c] = max(distance[c], r)
    # normalize
    distance /= np.amax(distance)

    cardinal_map[iframe] = pd.DataFrame({'distance': distance})

com = pd.DataFrame({'cx': cxs, 'cy': cys})
com.to_pickle("data/ruffling_2d/com.pkl")

cardinal_map = pd.concat(cardinal_map.values(), axis=0, keys=cardinal_map.keys())
cardinal_map.to_pickle("data/ruffling_2d/cardinal_map.pkl")
