import boltons.debugutils
#boltons.debugutils.pdb_on_exception()

from vispy import app, scene
from vispy.color import get_colormap

import numpy as np

from utoolbox.utils.decorators import timeit
from utoolbox.viewers.vispy.volume import MultiVolume

file_path = 'data/membrane.tif'

@timeit
def load_file():
    import utoolbox.io.primitives as dtype
    return dtype.SimpleVolume(file_path)

data = load_file()

# create vispy environment
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# attach data to visual node
vol_node = MultiVolume([data], cmaps=['GrBu'], method='mip', max_vol=1,
                       parent=view.scene)
vol_node.transform = scene.STTransform(translate=(0, 0, 0))

view.camera = scene.cameras.TurntableCamera(parent=view.scene, fov=60.)
# flip z-axis
view.camera.flip = (False, False, True)

canvas.update()

if __name__ == '__main__':
    app.run()
