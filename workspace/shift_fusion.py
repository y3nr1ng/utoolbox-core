import logging

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

from utoolbox.transform.projections import Orthogonal
from utoolbox.utils.decorator import timeit

logger = logging.getLogger(__name__)


@timeit
def create_mips(vol):
    # XY, XZ, YZ
    ortho = Orthogonal(vol)
    return ortho.xy, ortho.xz, ortho.yz

    # return vol.max(axis=0), vol.max(axis=1), vol.max(axis=2)


@timeit
def pairwise_shift(mips_a, mips_b):
    shifts = tuple(
        register_translation(im_a, im_b, upsample_factor=10)[0]
        for im_a, im_b in zip(mips_a, mips_b)
    )
    logger.info(f"shifts, YX={shifts[0]}, ZX={shifts[1]}, ZY={shifts[2]}")
    return shifts


def preview_shifts(mips_a, mips_b, shfits):
    for mip_a, mip_b, shift in zip(mips_a, mips_b, shfits):
        fig = plt.figure()
        ax_a = plt.subplot(1, 3, 1)
        ax_a.imshow(mip_a, cmap="gray")
        ax_a.set_axis_off()
        ax_a.set_title("Reference")

        ax_b = plt.subplot(1, 3, 2)
        ax_b.imshow(mip_b, cmap="gray")
        ax_b.set_axis_off()
        ax_b.set_title("Target")

        offset_b = fourier_shift(np.fft.fftn(mip_b), shift)
        offset_b = np.fft.ifftn(offset_b).real
        summed = mip_a + offset_b
        ax_ab = plt.subplot(1, 3, 3)
        ax_ab.imshow(summed, cmap="gray")
        ax_ab.set_axis_off()
        ax_ab.set_title("Summed")


def main():
    paths = [
        "fusion/bead_offset20um_zp1um_501_overall_PSF_NAp30nap26_150ms_2.tif",
        "fusion/bead_offsetn20um_zp1um_501_overall_PSF_NAp30nap26_150ms_2.tif",
    ]

    logger.info("create MIPs")
    mips = []
    for path in paths:
        logger.debug(f".. {path}")
        im = imageio.volread(path)
        mips.append(create_mips(im))

    logger.info("determine shifts between axes")
    shifts = pairwise_shift(mips[0], mips[1])
    # preview_shifts(mips[0], mips[1], shifts)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
    plt.show()
