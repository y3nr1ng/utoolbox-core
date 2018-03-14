import os

import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
import imageio

from utoolbox.container import Volume
from utoolbox.analysis.roi import extract_mask, mask_to_contour


source_folder = "data/RFiSHp2aLFCYC/decon/488"

def load_data(root):
    files = []
    for fname in os.listdir(root):
        #TODO more sophisticate filename filter
        if fname.endswith(".tif"):
            files.append(os.path.join(root, fname))
    files = files[0:5]
    print("{} data founded under \"{}\"".format(len(files), root))

    #TODO sort by timestamp

    for index, fname in enumerate(files):
        print("[{}] {}".format(index, os.path.basename(fname)))
        data = Volume(fname)
        data = rescale_intensity(data, out_range=(0., 1.))

        # XY MIP
        data = np.amax(data, axis=0)

        yield data

masks = extract_mask(load_data(source_folder), iterative=False)

contours = mask_to_contour(masks)
points = {}
for index, contour in enumerate(contours):
    imageio.imwrite(
        "data/ruffling_2d/contour_{:03}.tif".format(index),
        rescale_intensity(contour.astype(np.int32))
    )

    indices = np.where(contour == True)
    points[index] = pd.DataFrame({'x': indices[1], 'y': indices[0]})

points = pd.concat(points.values(), axis=0, keys=points.keys())
points.to_pickle("data/ruffling_2d/contours.pkl")
