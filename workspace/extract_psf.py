import logging
import os

import imageio
import numpy as np

from vispy import app, scene
from vispy.color.colormap import BaseColormap, Colormap
from vispy.visuals.transforms import STTransform

from utoolbox.analysis.psf_average import PSFAverage
from utoolbox.data.datastore import FolderDatastore

logger = logging.getLogger(__name__)


def preview_volume(vol, scale=(0.101, 0.101, 0.1)):
    canvas = scene.SceneCanvas(keys="interactive")
    canvas.size = 1024, 1024
    canvas.show()

    # create view box
    view = canvas.central_widget.add_view()

    # genereate colormap
    
    n_colors = 256
    alphas = np.linspace(0., 1., n_colors)
    color = np.c_[
        np.zeros((n_colors)), np.ones((n_colors,)), np.ones((n_colors,)), alphas
    ]
    cmap2 = Colormap(color)

    
    class TransGrays(BaseColormap):
        glsl_map = """
        vec4 translucent_grays(float t) {
            return vec4(t, t, t, t*0.05);
        }
        """
    cmap = TransGrays()

    """
    from utoolbox.data.io.amira import AmiraColormap

    color = AmiraColormap("volrenGlow.am")
    color = np.array(color)
    print(color.shape)
    cmap = Colormap(color)
    """

    volume = scene.visuals.Volume(
        vol, cmap=cmap, parent=view.scene, clim=(5000, 50000), emulate_texture=False
    )
    volume.method = "translucent"

    # volume.transform = scene.STTransform(translate=(64, 64, 0))
    # set voxel scale
    # volume.transform = scene.STTransform(translate=shift)

    # assign camera
    camera = scene.cameras.TurntableCamera(parent=view.scene, fov=60.0, name="Arcball")
    view.camera = camera

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
    vol = imageio.volread('fusion_exm/sample9_zp6um_561_2.tif')
    vol = vol[:, ::4, ::4]
    print(vol.shape)

    import cupy as cp
    vol = cp.asarray(vol)
    from utoolbox.exposure import auto_contrast
    vol = auto_contrast(vol, auto_threshold=100000000)
    vol = cp.asnumpy(vol)
    print(vol.dtype)

    preview_volume(vol)
    
    #main()
