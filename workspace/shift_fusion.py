import logging
from math import ceil, floor

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import (
    register_translation,
    corner_harris,
    corner_peaks,
    corner_subpix,
)
from skimage.transform import EuclideanTransform
from skimage.measure import ransac
from scipy.ndimage import fourier_shift

from vispy import app, scene
from vispy.color.colormap import Colormap

from utoolbox.data.datastore import FolderDatastore
from utoolbox.feature import DftRegister
from utoolbox.stitching import Sandbox
from utoolbox.stitching.phasecorr import PhaseCorrelation
from utoolbox.transform.projections import Orthogonal
from utoolbox.util.decorator import timeit

logger = logging.getLogger(__name__)


def preview_shifts(a, b, shifts):

    canvas = scene.SceneCanvas(keys="interactive")
    canvas.size = 1024, 1024
    canvas.show()

    # create view box
    vb_xy = scene.widgets.ViewBox(border_color="white", parent=canvas.scene)
    vb_xz = scene.widgets.ViewBox(border_color="white", parent=canvas.scene)
    vb_yz = scene.widgets.ViewBox(border_color="white", parent=canvas.scene)
    vbs = vb_xy, vb_xz, vb_yz

    # put them in a grid
    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    grid.add_widget(vb_xy, 0, 0)
    grid.add_widget(vb_xz, 1, 0)
    grid.add_widget(vb_yz, 0, 1)

    # genereate colormap
    n_colors = 128
    alphas = np.linspace(0.0, 1.0, n_colors)
    color_red = np.c_[
        np.ones((n_colors,)), np.zeros((n_colors,)), np.zeros((n_colors)), alphas
    ]
    cmap_red = Colormap(color_red)
    color_blue = np.c_[
        np.zeros((n_colors)), np.zeros((n_colors,)), np.ones((n_colors,)), alphas
    ]
    cmap_blue = Colormap(color_blue)

    # build shifts for mips
    sz, sy, sx = shifts
    nz, ny, nx = a.shape
    shifts = ((sx, sy), (sx, sz), (sy, sz))
    print(shifts)

    # create visuals
    i = 0
    for im, cm in zip((a, b), (cmap_red, cmap_blue)):
        mips = [im.max(axis=axis) for axis in range(3)]
        for vb, mip, shift in zip(vbs, mips, shifts):
            image = scene.visuals.Image(mip, cmap=cm, parent=vb.scene)
            image.set_gl_state("translucent", depth_test=False)

            # apply transformation
            if i > 0:
                image.transform = scene.STTransform(translate=shift)
            else:
                i += 1

    # assign cameras
    for vb in vbs:
        vb.camera = scene.PanZoomCamera(aspect=1)
        vb.camera.set_range()
        vb.camera.flip = (0, 1, 0)

    app.run()


def main():
    ds = FolderDatastore("fusion_psf", read_func=imageio.volread, extensions=["tif"])
    logger.info(f"{len(ds)} file(s) found")

    # sandbox = Sandbox(ds)

    ratio = (2, 8, 8)
    sampler = tuple([slice(None, None, r) for r in ratio])

    overlap = 0.5

    tiles = list(ds.keys())
    for ref_key in tiles[:-1]:
        logger.debug(f".. loading {ref_key}")
        ref_im = ds[ref_key]
        # ref_im = ref_im[:, 1, :, :]
        ref_im_lo = ref_im[sampler]
        imageio.volwrite("ref_lo.tif", ref_im_lo)  # DEBUG
        print(ref_im_lo.shape)

        for tar_key in tiles[1:]:
            logger.info(f"{ref_key} -> {tar_key}")

            logger.debug(f".. loading {tar_key}")
            tar_im = ds[tar_key]
            # tar_im = tar_im[:, 1, :, :]
            tar_im_lo = tar_im[sampler]
            imageio.volwrite("tar_lo.tif", tar_im_lo)  # DEBUG

            pc = PhaseCorrelation(ref_im_lo, tar_im_lo)
            pc.run()
            


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
