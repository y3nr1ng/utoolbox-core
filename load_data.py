import boltons.debugutils
boltons.debugutils.pdb_on_exception()

from vispy import app, scene
from vispy.color import get_colormap

import numpy as np

from utoolbox.utils.decorators import timeit
from utoolbox.viewers.volume import MultiVolume

file_path = 'data/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif'

@timeit
def load_file():
    import utoolbox.io.primitives as dtype
    return dtype.SimpleVolume(file_path)

data = load_file()

# create vispy environment
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# attach data to visual node
volumes = [
    (data, None, get_colormap('GrBu'))
]
vol_node = MultiVolume(volumes, parent=view.scene)
vol_node.transform = scene.STTransform(translate=(64, 64, 0))

view.camera = scene.cameras.TurntableCamera(parent=view.scene, fov=60.)

canvas.update()

if __name__ == '__main__':
    app.run()
