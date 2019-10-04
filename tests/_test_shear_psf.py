import imageio
import matplotlib.pyplot as plt
import numpy as np

import utoolbox.simulate.psf as psf

parms = psf.FastGibsonLanni.Parameters(
    M=68,     # magnification
    NA=1.1,   # numerical aperture
    ni0=1.33, # immersion medium refraction index, design value
    ni=1.33,  # immersion medium refraction index, experimental value
    ns=1.33,  # specimen refractive index
    ti0=100,  # working distance [um]
)

angle = 32.8
img_res = (0.102, 0.25)
model = psf.ShearedPSF(
    psf.FastGibsonLanni,
    angle, img_res,
    parms, has_coverslip=False
)

psf = model((256, 512, 512), 0.488)
print(psf.shape)
imageio.volwrite("psf.tif", psf)
