from queue import Queue

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.segmentation import mark_boundaries


# formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='CBC Group'), bitrate=1800)

fig = plt.figure()

points = pd.read_pickle("data/ruffling_2d/contours.pkl")
com = pd.read_pickle("data/ruffling_2d/com.pkl")

queue = Queue(maxlen=3)
contour = np.zeros((810, 810), dtype=np.uint8)
for iframe in points.index.levels[0]:
    x = points.loc[iframe, 'x'].values
    y = points.loc[iframe, 'y'].values

    # reconstruct contour
    contour = np.zeros_like(canvas)
    contour[y, x] = 255

    plt.imshow(contour, cmap='binary')
    plt.scatter(com[iframe, 'x'], com[iframe, 'y'], s=250, c='red', marker='+')

    line_ani = animation.FuncAnimation(fig, update_line, 25, fargs=(data, l),
                                   interval=50, blit=True)
    line_ani.save('lines.mp4', writer=writer)
