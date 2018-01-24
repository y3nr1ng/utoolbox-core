"""
Example volume rendering
Controls:
* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold

With fly camera:
* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around
"""
from itertools import cycle

import numpy as np

from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform

from utoolbox.io import TimeSeries
import utoolbox.io.primitives as dtype
from utoolbox.utils.decorators import timeit

file_path = 'data/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif'

# Read volume
data = dtype.SimpleVolume(file_path)

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', size=(810, 810), show=True)
canvas.measure_fps()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Create the volume visuals, only one is visible
volume = scene.visuals.Volume(data, parent=view.scene, threshold=0.225,
                              emulate_texture=False)
#volume.transform = scene.STTransform(translate=(0, 0, 0))

# Create three cameras (Fly, Turntable and Arcball)
fov = 0.
camera = scene.cameras.TurntableCamera(parent=view.scene, fov=fov,
                                     name='Turntable')
view.camera = camera

# Create an XYZAxis visual
axis = scene.visuals.XYZAxis(parent=view)
s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """

class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """

# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)


# Implement axis connection with cam2
@canvas.events.mouse_move.connect
def on_mouse_move(event):
    if event.button == 1 and event.is_dragging:
        axis.transform.reset()

        axis.transform.rotate(cam2.roll, (0, 0, 1))
        axis.transform.rotate(cam2.elevation, (1, 0, 0))
        axis.transform.rotate(cam2.azimuth, (0, 1, 0))

        axis.transform.scale((50, 50, 0.001))
        axis.transform.translate((50., 50.))
        axis.update()


# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global opaque_cmap, translucent_cmap
    if event.text == '2':
        methods = ['mip', 'translucent', 'iso', 'additive']
        method = methods[(methods.index(volume.method) + 1) % 4]
        print("Volume render method: %s" % method)
        cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
        volume.method = method
        volume.cmap = cmap
    elif event.text == '3':
        if volume.method in ['mip', 'iso']:
            cmap = opaque_cmap = next(opaque_cmaps)
        else:
            cmap = translucent_cmap = next(translucent_cmaps)
        volume.cmap = cmap
    elif event.text != '' and event.text in '[]':
        s = -0.025 if event.text == '[' else 0.025
        volume.threshold += s
        th = volume.threshold
        print("Isosurface threshold: %0.3f" % th)

# for testing performance
# @canvas.connect
# def on_draw(ev):
# canvas.update()

if __name__ == '__main__':
    print(__doc__)
    app.run()
