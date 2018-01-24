from vispy import scene, visuals

class VolumeRender(object):
    """Simple volume renderer."""

    def __init__(self, title=None, size=(800, 600)):
        self.canvas = scene.SceneCanvas(title=title,
                                        keys='interactive', # remap F11
                                        size=size,
                                        show=True,          # show immediately
                                        data=None)
        self.view = self.canvas.central_widget.add_view()
        self._attach_camera()
        self._attach_axis()
        if data:
            self._attach_volume(data)

    def _attach_camera(self, fov=50):
        self.camera = scene.cameras.TurntableCamera(parent=self.view.scene,
                                                    fov=fov,
                                                    name='Turntable')
        self.view.camera = self.camera

    def _attach_axis(self):
        self.axis = scene.visuals.XYZAxis(parent=self.view)
        transform = visuals.STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
        self.axis.transform = transform.as_matrix()

    def _attach_volume(self, data, threshold=0.5):
        self.volume = scene.visuals.Volume(data,
                                           parent=self.view.scene,
                                           threshold=threshold,
                                           emulate_texture=False)

    def show(self):
        self.canvas.show()

    def update_data(self, data):
        if data:
            self.volume.set_data(data)
        else:
            self._attach_volume(data)
