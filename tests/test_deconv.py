import logging

import coloredlogs
import imageio
import numpy as np
import pyopencl as cl

from utoolbox.deconvolve import RichardsonLucy
import utoolbox.parallel
import utoolbox.simulate.psf as psf

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####
#path = "../data/cell4_488nm_crop.tif"
path = "raw/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs.tif"
I_in = imageio.volread(path)


##### GENERATE PSF MODEL #####
parms = psf.FastGibsonLanni.Parameters(
    M=68,     # magnification
    NA=1.1,   # numerical aperture
    ni0=1.33, # immersion medium refraction index, design value
    ni=1.33,  # immersion medium refraction index, experimental value
    ns=1.33,  # specimen refractive index
    ti0=100,  # working distance [um]
)

angle = 32.8
img_res = (0.102, 0.5)
model = psf.ShearedPSF(
    psf.FastGibsonLanni,
    angle, img_res,
    parms, has_coverslip=False
)

psf = model(I_in.shape, 0.488)
imageio.volwrite("psf.tif", psf)
logger.info("psf generated, shape={}".format(psf.shape))


##### EXCEUTE DECONVOLUTION #####
ctx = utoolbox.parallel.create_some_context(dev_type='gpu', vendor='NVIDIA')
with RichardsonLucy(ctx, I_in.shape, prefer_add=False, n_iter=10) as model:
    model.psf = psf
    I_out = model(I_in)


##### RESULT #####
imageio.volwrite("result.tif", I_out)
