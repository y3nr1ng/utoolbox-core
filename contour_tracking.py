import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_pickle("data/ruffling_2d/contours.pkl")

n_bins = 24
bins = np.linspace(-np.pi, np.pi, n_bins+1, dtype=np.float)
pol_grid = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)

# create plots
fig = plt.figure()
fig.subplots_adjust(wspace=0.5)
ax_d = fig.add_subplot(1, 2, 1, projection='polar')
ax_d.set_title('Relative Extents', y=1.2)
ax_d.set_theta_direction(-1)
ax_d.set_theta_zero_location('E')
bar_d = ax_d.bar(pol_grid, np.zeros_like(pol_grid), width=2*np.pi/n_bins, color='lightblue', edgecolor='k')
ax_d.set_yticks([0, 1])

ax_v = fig.add_subplot(1, 2, 2, projection='polar')
ax_v.set_title('Unit Velocity', y=1.2)
ax_v.set_theta_direction(-1)
ax_v.set_theta_zero_location('E')
bar_v = ax_v.bar(pol_grid, np.zeros_like(pol_grid), width=2*np.pi/n_bins, color='lightblue', edgecolor='k')
ax_v.set_yticks([-1, 0, 1])

cxs = []
cys = []
distance = np.zeros(n_bins, dtype=np.float)
velocity = np.zeros_like(distance)
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
    distance.fill(0)
    for c, r in zip(cardinal, radius):
        distance[c] = max(distance[c], r)
    # normalize
    rel_distance = distance / np.amax(distance)

    data = {'distance': distance}

    if iframe > 0:
        velocity = distance - cardinal_map[iframe-1]['distance'].values
        velocity /= np.amax(velocity)
        data['velocity'] = velocity
    rel_velocity = velocity / np.amax(velocity)
    np.nan_to_num(rel_velocity, copy=False)

    # save to DataFrame
    cardinal_map[iframe] = pd.DataFrame(data)

    # save plot
    for obj, height in zip(bar_d, rel_distance):
        obj.set_height(height)
    for obj, height in zip(bar_v, rel_velocity):
        obj.set_height(height)
    fig.canvas.draw()

    fig.savefig("data/ruffling_2d/cardinal_{:03}.tif".format(iframe))

com = pd.DataFrame({'cx': cxs, 'cy': cys})
com.to_pickle("data/ruffling_2d/com.pkl")

cardinal_map = pd.concat(cardinal_map.values(), axis=0, keys=cardinal_map.keys())
cardinal_map.to_pickle("data/ruffling_2d/cardinal_map.pkl")
