import logging
import os

import coloredlogs
import imageio
import numpy as np
import pycuda.driver as cuda

from utoolbox.container import ImplTypes
from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform.deskew import Deskew

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

ctx = create_some_context()
ctx.push()

rotate = Deskew(ImplTypes.GPU)

I = imageio.volread(
    'cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs.tif'
)
J = rotate(I, 32.8, res=(.102, .5), rotate=True, resample=False)
imageio.imwrite('output.tif', J)

cuda.Context.pop()
