import glob
import logging
import os

import coloredlogs
import imageio
import numpy as np
import pycuda.driver as cuda

from utoolbox.container import ImplTypes
from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform.rotate import Rotate2

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

ctx = create_some_context()
ctx.push()

rotate = Rotate2(ImplTypes.GPU)

I = imageio.imread('lena512.bmp')
J = np.empty((512, 256), dtype=np.float32)
rotate(I.astype(np.float32), 30., (1., 1.), J)
imageio.imwrite('output.bmp', J.astype(np.uint8))

cuda.Context.pop()