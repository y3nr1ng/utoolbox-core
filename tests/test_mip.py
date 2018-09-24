import logging
import os

import coloredlogs
import imageio
import numpy as np
import pycuda.driver as cuda
import tqdm

from utoolbox.utils.decorators import timeit
from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform import MIPTransform

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####
path = "deskew_output.tif"
I_in = imageio.volread(path)
logger.info("I_in.shape={}".format(I_in.shape))


##### EXCEUTE DESKEW #####
@timeit
def gpu(data):
    ctx = create_some_context()
    ctx.push()

    transform = MIPTransform('z')
    for _ in tqdm.trange(100):
        I_out = transform(I_in)

    cuda.Context.pop()

    return I_out

@timeit
def cpu(data):
    for _ in tqdm.trange(100):
        I_out = np.max(data, axis=0).squeeze()
    return I_out
    
I_out_gpu = gpu(I_in)
I_out_cpu = cpu(I_in)

##### RESULT #####
#imageio.imwrite("deskew_output_mip.tif", I_out)
