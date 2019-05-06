import logging

import coloredlogs
import imageio
import numpy as np 

from utoolbox.transform.deskew import Deskew

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

I = imageio.volread(
    'cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs.tif'
)

deskew_func = Deskew(rotate=False)
J = deskew_func.run(I)

#J = rotate(I, 32.8, res=(.102, .5), rotate=True, resample=False)
imageio.volwrite('output.tif', J)
