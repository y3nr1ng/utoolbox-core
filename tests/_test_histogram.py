import logging
import os

import coloredlogs
import numpy as np

from utoolbox.exposure.histogram import histogram
from utoolbox.util.decorator import timeit

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

shape = (256, 2048, 1025)
n_bins = 256

array = np.random.randint(0, 2**16-1, shape, np.uint16)

@timeit
def cpu():
    hist_cpu, _ = np.histogram(array, n_bins, (0, 2**16))
    return hist_cpu
hist_cpu = cpu()
#print(hist_cpu)
#print("n={}".format(hist_cpu.sum()))

@timeit
def gpu():
    return histogram(array, n_bins)
hist_gpu = gpu()
#print(hist_gpu)
#print("n={}".format(hist_gpu.sum()))

np.testing.assert_array_equal(hist_cpu, hist_gpu)
