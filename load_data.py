import boltons.debugutils
#boltons.debugutils.pdb_on_exception()

from vispy import app, scene, gloo
#from vispy.scene import Volume
from vispy.color import BaseColormap

import numpy as np

from utoolbox.utils.decorators import timeit
from utoolbox.viewers.vispy.volume import MultiVolume

membrane_path = 'data/membrane_000.tif'
cytosol_path = 'data/cytosol_000.tif'

@timeit
def load_file(file_path):
    import utoolbox.io.primitives as dtype
    return dtype.SimpleVolume(file_path)

membrane_data = load_file(membrane_path)
print("(min, max) = ({}, {})".format(np.amin(membrane_data), np.amax(membrane_data)))
cytosol_data = load_file(cytosol_path)

# create vispy environment
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

def get_translucent_cmap(r, g, b, name="untitled"):
    class Translucent(BaseColormap):
        glsl_map = """
        vec4 translucent_{name}(float t)
        {{
            return vec4(t*{0}, t*{1}, t*{2}, t*0.5);
        }}
        """.format(r, g, b, name=name)
    return Translucent()

# attach data to visual node
vol_node = MultiVolume([membrane_data, cytosol_data], clims=[(100, 2560), (100, 2560)],
                       cmaps=[get_translucent_cmap(0, 1, 0, "green"), get_translucent_cmap(0, 0, 1, "blue")],
                       method='additive', max_vol=2, parent=view.scene)
vol_node.transform = scene.STTransform(translate=(0, 0, 0))

#vol_node_1 = Volume(membrane_data, clim=(0, 756), cmap='grays', method='mip', parent=view.scene)
#vol_node_1.transform = scene.STTransform(translate=(0, 0, 0))

#vol_node_2 = Volume(cytosol_data, clim=(0, 756), cmap='grays', method='mip', parent=view.scene)
#vol_node_2.transform = scene.STTransform(translate=(0, 0, 0))

view.camera = scene.cameras.TurntableCamera(parent=view.scene, fov=60.)
# flip z-axis
view.camera.flip = (False, False, True)

canvas.update()
canvas.measure_fps()

if __name__ == '__main__':
    """
    viewer = app.Application()
    print(viewer.native)
    from PyQt5.QtCore import Qt
    print(canvas.central_widget)
    canvas.central_widget.setAttribute(Qt.WA_NoSystemBackground, False)
    viewer.run()
    """
    app.run()
