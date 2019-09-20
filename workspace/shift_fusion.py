import logging

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

from vispy import app, scene
from vispy.color.colormap import Colormap

from utoolbox.data.datastore import FolderDatastore
from utoolbox.feature import DftRegister
from utoolbox.stitching import Sandbox
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
    def dft_register(a, b):
        with DftRegister(a, upsample_factor=8) as func:
            return func.register(b, return_error=True)

    shifts, errors = zip(
        *(
            # register_translation(im_a, im_b, upsample_factor=8)[:2]
            dft_register(im_a, im_b)
            for im_a, im_b in zip(mips_a, mips_b)
        )
    )
    logger.info(
        f"shifts, YX={shifts[0]} ({errors[0]:.4f}), ZX={shifts[1]} ({errors[1]:.4f}), ZY={shifts[2]} ({errors[2]:.4f})"
    )
    return shifts, errors


def preview_shifts(mips_a, mips_b, shfits):
    for mip_a, mip_b, shift in zip(mips_a, mips_b, shfits):
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
    ds = FolderDatastore("fusion", read_func=imageio.volread, extensions=["tif"])
    logger.info(f"{len(ds)} file(s) found")

    sandbox = Sandbox(ds)

    tiles = list(ds.keys())
    # pairwise
    mips = dict()
    for reference in tiles[:-1]:
        # reference
        try:
            reference_mip = mips[reference]
        except KeyError:
            reference_mip = create_mips(ds[reference])
            mips[reference] = reference_mip

        for target in tiles[1:]:
            if reference == target:
                continue
            logger.info(f"{reference} -> {target}")
            # target
            try:
                target_mip = mips[target]
            except KeyError:
                target_mip = create_mips(ds[target])
                mips[target] = target_mip

            # determine shift
            shifts, errors = pairwise_shift(reference_mip, target_mip)
            sandbox.link(reference, target, shifts, errors)

            canvas = scene.SceneCanvas(keys="interactive")
            canvas.size = 1024, 1024
            canvas.show()

            # Set up a viewbox to display the image with interactive pan/zoom
            view = canvas.central_widget.add_view()

            n_colors = 128
            alphas = np.linspace(0.0, 1.0, n_colors)

            color_red = np.c_[
                np.ones((n_colors,)),
                np.zeros((n_colors,)),
                np.zeros((n_colors)),
                alphas,
            ]
            cmap_red = Colormap(color_red)
            image_a = scene.visuals.Image(
                reference_mip[2], cmap=cmap_red, parent=view.scene
            )
            image_a.set_gl_state("translucent", depth_test=False)

            color_blue = np.c_[
                np.zeros((n_colors)),
                np.zeros((n_colors,)),
                np.ones((n_colors,)),
                alphas,
            ]
            cmap_blue = Colormap(color_blue)
            image_b = scene.visuals.Image(
                target_mip[2], cmap=cmap_blue, parent=view.scene
            )
            image_b.set_gl_state("translucent", depth_test=False)

            #image_b.transform = scene.STTransform(translate=shifts[0][::-1])

            dy, dx = shifts[2]
            ny, nx = target_mip[2].shape
            image_b.transform = scene.STTransform(translate=(dx, dy-ny))

            # dy, dx = shifts[2]
            #ny, nx = target_mip[2].shape
            #image_b.transform = scene.STTransform(translate=(shifts[0][1], dy - ny))

            # Set 2D camera (the camera will scale to the contents in the scene)
            view.camera = scene.PanZoomCamera(aspect=1)
            view.camera.set_range()
            view.camera.flip = (0, 1, 0)

            app.run()

    sandbox.update()


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
