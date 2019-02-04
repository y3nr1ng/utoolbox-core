import glob
import logging
import os

import coloredlogs
import imageio
import numpy as np
import pycuda.driver as cuda

from utoolbox.container import ImplTypes
from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform.shear import Shear2

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

ctx = create_some_context()
ctx.push()

shear = Shear2(ImplTypes.GPU)

I = imageio.imread('lena512.bmp')
J = shear(I, (.5, 0), scale=(.5, 1.)) #, shape=(512, 256))
imageio.imwrite('output.bmp', J)

cuda.Context.pop()