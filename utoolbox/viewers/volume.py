"""
3-D data viewer.

Example
-------
    viewer = VolumeViewer()
    viewer.show(run=True)
"""
from vispy import scene, gloo
from vispy.app import Timer
from vispy.scene import cameras

from utoolbox.viewers.vispy.volume import MultiVolume

class VolumeViewer(scene.SceneCanvas):
    def __init__(self, dataset, title='Volume Viewer',
                 size=(800, 600)):
        super(VolumeViewer, self).__init__(title=title, size=size)
        gloo.set_clear_color(color='black')

        self.unfreeze()

        self._view = self.central_widget.add_view()

        dataset = list(map(list, zip(*dataset)))
        self._volume = MultiVolume(
            dataset[0], clims=dataset[1], cmaps=dataset[2],
            method='additive', max_vol=2, parent=self._view.scene,
        )
        #shape = dataset[0][0].shape
        #self._volume.transform = scene.STTransform(translate=(-shape[0]/2, -shape[1]/2, -shape[2]/2))

        camera = cameras.TurntableCamera(parent=self._view.scene, fov=60., azimuth=0., elevation=60.)
        # flip z-axis
        camera.flip = (False, False, True)
        self._view.camera = camera

        self._timer = Timer('auto', connect=self.on_timer, start=True)

        self.freeze()

    def on_timer(self, event):
        #NOTE (az, el)
        self._view.camera.orbit(1., 0.)
        self.update()
