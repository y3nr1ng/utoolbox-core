import logging
import os

import imageio
import numpy as np

from vispy import app, scene
from vispy.color.colormap import BaseColormap, Colormap, ColorArray
from vispy.visuals.transforms import STTransform

from utoolbox.analysis.psf_average import PSFAverage
from utoolbox.data.datastore import FolderDatastore

logger = logging.getLogger(__name__)


def preview_volume(vols, shifts=None):
    canvas = scene.SceneCanvas(keys="interactive")
    canvas.size = 1024, 1024
    canvas.show()

    # create view box
    view = canvas.central_widget.add_view()

    # genereate colormap

    """
    n_colors = 256
    alphas = np.linspace(0.0, 1.0, n_colors)
    color = np.c_[
       alphas, alphas, alphas, alphas
    ]
    cmap = Colormap(color)
    """

    from utoolbox.data.io.amira import AmiraColormap

    color = AmiraColormap("volrenGlow.am")
    color = np.array(color)
    color[0, :] = 0
    color[:, 3] /= 100
    cmap = Colormap(color)

    for i, vol in enumerate(vols):
        volume = scene.visuals.Volume(
            vol, cmap=cmap, clim=(600, 3000), parent=view.scene, emulate_texture=False
        )
        volume.method = "translucent"
        volume.transform = scene.STTransform(scale=(2, 2, 5.5))

        volume.set_gl_state("translucent", depth_test=False)

        if shifts:
            volume.transform = scene.STTransform(translate=shifts[i])

    # assign camera
    camera = scene.cameras.TurntableCamera(parent=view.scene, fov=60.0, name="Arcball", elevation=30.)
    view.camera = camera
    view.camera.flip = (False, True, True)

    view.camera.reset()

    # axis
    axis = scene.visuals.XYZAxis(parent=view)
    s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
    affine = s.as_matrix()
    axis.transform = affine
    # link with camera
    @canvas.events.mouse_move.connect
    def on_mouse_move(event):
        if event.button == 1 and event.is_dragging:
            axis.transform.reset()

            axis.transform.rotate(camera.roll, (0, 0, 1))
            axis.transform.rotate(camera.elevation, (1, 0, 0))
            axis.transform.rotate(camera.azimuth, (0, 1, 0))

            axis.transform.scale((50, 50, 0.001))
            axis.transform.translate((50.0, 50.0))
            axis.update()

    # render rotation movie
    """
    n_steps = 240
    axis = [0, 0, 0]

    logger.debug(".. rendering")
    step_angle = 360.0 / n_steps
    writer = imageio.get_writer("t1-head_split_rotate.mp4", fps=24)
    for i in range(n_steps):
        im = canvas.render()
        writer.append_data(im)
        view.camera.transform.rotate(step_angle, axis)
    writer.close()
    """

    app.run()


def main(root="fusion_psf"):
    ds = FolderDatastore(root, read_func=imageio.volread, extensions=["tif"])
    logger.info(f"{len(ds)} file(s) found")

    for key, vol in ds.items():
        logger.info(key)

        dst_dir = os.path.join(root, key)
        try:
            os.mkdir(dst_dir)
        except:
            # folder exists
            pass

        psf_avg = PSFAverage((97, 97, 97))
        psfs = psf_avg(vol, return_coords=True)

        psf_average = None
        for i, (sample, coord) in enumerate(psfs):
            coord = [f"{c:04d}" for c in reversed(coord)]
            coord = "-".join(coord)
            fname = f"psf_{i:03d}_{coord}.tif"
            imageio.volwrite(os.path.join(dst_dir, fname), sample)

            try:
                psf_average = (psf_average + sample) / 2
            except TypeError:
                psf_average = sample

        import cupy as cp

        psf_average = cp.asarray(psf_average)
        from utoolbox.exposure import auto_contrast

        psf_average = auto_contrast(psf_average)
        psf_average = cp.asnumpy(psf_average)

        preview_volume(psf_average)

        break


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    """
    from vispy import io

    vol = np.load(io.load_data_file("brain/mri.npz"))["data"]
    print(vol.dtype)
    """

    import imageio
    vol = imageio.volread('20181019_expanded_hippo/1-Pos_002_005.tif')

    import cupy as cp

    vol = cp.asarray(vol)
    from utoolbox.exposure import auto_contrast

    vol = auto_contrast(vol)
    vol = cp.asnumpy(vol)
    #vol = np.swapaxes(vol, 0, 1)
    print(vol.dtype)

    """
    avg, std = vol.mean(), vol.std()
    vol[vol < (avg - std)] = 0

    nz, ny, nx = vol.shape
    mid = ny // 2
    vol1 = vol[:, :mid, :]
    vol2 = vol[:, mid:, :]

    preview_volume((vol1, vol2), ((0, -mid, 0), (0, mid, 0)))
    """

    vol = vol[:, ::2, ::2]
    preview_volume((vol, ))
    # main()

