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

model = psf.FastGibsonLanni(parms, resolution=(0.102, 0.1), has_coverslip=False)

psf = model((256, 512, 512), 0.488)
print(psf.shape)
imageio.volwrite("psf.tif", psf)

"""
rhov, phase, J, C = model((128, 256, 256), 0.590)

# z-plane to compute
z0 = 24

# The Fourier-Bessel approximation
est = J.T.dot(C[:,z0])

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
ax[0].plot(rhov, np.real(phase[z0, :]), label=r'$ \exp{ \left[ jW \left( \rho \right) \right] }$')
ax[0].plot(rhov, np.real(est), '--', label='Fourier-Bessel')
ax[0].set_xlabel(r'$\rho$')
ax[0].set_title('Real')
ax[0].legend(loc='upper left')

ax[1].plot(rhov, np.imag(phase[z0, :]))
ax[1].plot(rhov, np.imag(est), '--')
ax[1].set_xlabel(r'$\rho$')
ax[1].set_title('Imaginary')
ax[1].set_ylim((-1.5, 1.5))

plt.show()
"""
