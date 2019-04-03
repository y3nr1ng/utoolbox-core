import logging
from math import hypot
import os

import coloredlogs
import imageio
import numpy as np
from tqdm import tqdm

from skimage.feature import register_translation

from utoolbox.container.datastore import ImageDatastore
from utoolbox.feature import DftRegister
from utoolbox.util.logging import TqdmLoggingHandler

from utoolbox.util.decorator import timeit

###
# region: Configure logging facilities
###

logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())

logging.getLogger('tifffile').setLevel(logging.ERROR)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

###
# endregion
###

def tuple_float_to_str(t, digits=4):
    float_fmt = '{{:.{}f}}'.format(digits)
    fmt = '(' + ', '.join([float_fmt, ] * len(t)) + ')'
    return fmt.format(*list(t))

## 
# region: DEBUG test register
##
#template_path = 'data/xy/cell7_zp3um_561_1.tif'
#target_path = 'data/xy/cell7_zp3um_561_9.tif'
#
#template = imageio.imread(template_path)
#template = np.asarray(template)
#
#target = imageio.imread(target_path)
#target = np.asarray(target)
#
#
#@timeit
#def gpu():
#    with DftRegister(template, upsample_factor=100) as dft_reg:
#        for _ in range(100):
#            shifts, error = dft_reg.register(target, return_error=True)
#    return shifts, error
#shifts, error = gpu()
#logger.info("gpu, shifts={}, error={:.4f}".format(tuple_float_to_str(shifts), error))
#
#@timeit
#def cpu():
#    for _ in range(100):
#        shifts, error, _ = register_translation(
#            template, target, upsample_factor=100
#        )
#    return shifts, error
#shifts, error = cpu()
#logger.info("cpu, shifts={}, error={:.4f}".format(tuple_float_to_str(shifts), error))
## 
# endregion
##

data_dir = 'data'
projs = ('xy', 'xz', 'yz')

imds = ImageDatastore(
    #os.path.join(data_dir, 'xy'),
    "/Users/Andy/Desktop/mouse_2/slice_1/grayscale",
    imageio.volread
)
from pprint import pprint
pprint(list(zip(enumerate(imds.files))))
im_projs = list(imds)

im = im_projs[0]
logger.info("shape={}, {}".format(im.shape, im.dtype))

# upsampling factor
uf = 10
# overlap ratio
overlap_perct = 0.5

shift_min = min(im.shape) * (1.-overlap_perct)
shift_max = hypot(*im.shape) * (1.+overlap_perct)

logger.info("overlap percentage {:.1f}%, shifts [{:.2f}, {:.2f}]".format(
    overlap_perct*100, shift_min, shift_max
))

from math import hypot
@timeit
def gpu():
    n_im = len(imds)
    graph = {}
    for i_ref in range(n_im):
        with DftRegister(im_projs[i_ref], upsample_factor=uf) as dft_reg:
            adj_list = []
            for i_tar in range(i_ref+1, n_im):
                shifts, error = dft_reg.register(
                    im_projs[i_tar], return_error=True
                )
                sy, sx = shifts

                d = hypot(*shifts)
                print(d)
                if (d < shift_min) or (d > shift_max):
                    continue
                adj_list.append([i_tar, error, shifts, hypot(shifts[0], shifts[1])])
                #adj_list.append([i_tar, error])
            graph[i_ref] = adj_list
    return graph
graph = gpu()

from pprint import pprint
pprint(graph)

#raise RuntimeError("DEBUG")
@timeit
def cpu():
    n_im = len(imds)
    graph = {}
    for i_ref in range(n_im):
        adj_list = []
        for i_tar in range(i_ref+1, n_im):
            shifts, error, _ = register_translation(
                im_projs[i_ref], im_projs[i_tar], upsample_factor=uf
            )
            adj_list.append([i_tar, error])
        graph[i_ref] = adj_list
    return graph
#graph = cpu()

#from pprint import pprint
#pprint(graph)

#raise RuntimeError("DEBUG")

# https://codereview.stackexchange.com/questions/174946/prims-algorithm-using-heapq-module-in-python
from heapq import heappush, heappop

def prims_algorithm(graph):
    explored = [] # vertices in tree
    start = next(iter(graph)) # arbitrary starting vertex
    #unexplored = [(0, start)] # unexplored edges, (cost, vertex) pairs
    unexplored = [(0, 9)]
    print(unexplored)
    while unexplored:
        cost, vertex = heappop(unexplored)
        if vertex not in explored:
            explored.append(vertex)
            for neighbor, cost, _, _ in graph[vertex]:
                if neighbor not in explored:
                    heappush(unexplored, (cost, neighbor))
        print()
        print(">> explored")
        pprint(explored)
        print(">> unexplored")
        pprint(unexplored)
    return explored

path = prims_algorithm(graph)
print()
print(' -> '.join([str(p) for p in path]))