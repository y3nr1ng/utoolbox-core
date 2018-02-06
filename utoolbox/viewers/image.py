"""
2-D data viewer.

# create the viewer and attach dataset
viewer = Canvas()
viewer.show()

# auto-start the application if in a standalone application
import sys
if sys.flags.interactive != 1:
    viewer.app.run()
"""
from vispy import app
from vispy.visuals import ImageVisual

class ImageViewer(app.Canvas):
    def __init__(self, data, size=None):
        #TODO calculate size that fit the screen
        super(ImageViewer, self).__init__(keys='interactive', size=size)
        self.image = ImageVisual(data, method='subdivide')

        self.show()

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        self.image.draw()

    def on_resize(self, event):
        pass
