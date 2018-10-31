import boltons.debugutils
#boltons.debugutils.pdb_on_exception()

from vispy import app, scene, gloo
from vispy.color import BaseColormap

import numpy as np

from utoolbox.util.decorator.benchmark import timeit
from utoolbox.viewers.volume import VolumeViewer

membrane_path = 'data/membrane_000.tif'
cytosol_path = 'data/cytosol_000.tif'

@timeit
def load_file(file_path):
    import utoolbox.io.primitives as dtype
    return dtype.SimpleVolume(file_path)

membrane_data = load_file(membrane_path)
cytosol_data = load_file(cytosol_path)

def get_translucent_cmap(r, g, b, name="untitled"):
    from vispy.color import BaseColormap
    class Translucent(BaseColormap):
        glsl_map = """
        vec4 translucent_{name}(float t)
        {{
            return vec4(t*{0}, t*{1}, t*{2}, 1.);
        }}
        """.format(r, g, b, name=name)
    return Translucent()

viewer = VolumeViewer([
    (membrane_data, (64, 3584), get_translucent_cmap(0, 1, 0, "green")),
    (cytosol_data, (32, 3584), get_translucent_cmap(0, 0, 1, "blue"))
])
viewer.show(run=True)
