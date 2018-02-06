"""
3-D data viewer.

# create the viewer and attach dataset
viewer = Canvas()
viewer.show()

# auto-start the application if in a standalone application
import sys
if sys.flags.interactive != 1:
    viewer.app.run()
"""
from vispy import app

class VolumeViewer(app.Canvas):
    def __init__(self):
        #TODO attach data

        #TODO attach camera

        #TODO determine update interval 'auto' or assigned
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_draw(self, ev):
        pass

    def on_resize(self, event):
        pass

    def on_timer(self, event):
        #TODO rotate the volume
        self.update()
