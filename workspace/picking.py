import numpy as np
from vispy import app, visuals, scene, color


def distance_traveled(positions):
    """
    Return the total amount of pixels traveled in a sequence of pixel
    `positions`, using Manhattan distances for simplicity.
    """
    return np.sum(np.abs(np.diff(positions, axis=0)))


def scatter3d(positions, colors, symbol='o', size=4.5, click_radius=2,
              on_click=None):
    """
    Create a 3D scatter plot window that is zoomable and rotateable, with
    markers of a given `symbol` and `size` at the given 3D `positions` and in
    the given RGBA `colors`, formatted as numpy arrays of size Nx3 and Nx4,
    respectively. Takes an optional callback function that will be called with
    the index of a clicked marker and a reference to the Markers visual
    whenever the user clicks a marker (or at most `click_radius` pixels next to
    a marker).
    """
    # based on https://github.com/vispy/vispy/issues/1189#issuecomment-198597473
    # create canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='gray')
    # create viewbox for user interaction
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = abs(positions).max() * 2.5
    # create visuals
    p1 = scene.visuals.Markers(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    #axis = scene.visuals.XYZAxis(parent=view.scene)
    # set positions and colors
    kwargs = dict(symbol=symbol, size=size, edge_color=None)
    p1.set_data(positions, face_color=colors, **kwargs)
    # prepare list of unique colors needed for picking
    ids = np.arange(1, len(positions) + 1, dtype=np.uint32).view(np.uint8)
    ids = ids.reshape(-1, 4)
    ids = np.divide(ids, 255, dtype=np.float32)
    # connect events
    if on_click is not None:
        def on_mouse_release(event):
            if event.button == 1 and distance_traveled(event.trail()) <= 2:
                # vispy has some picking functionality that would tell us
                # whether any of the scatter points was clicked, but we want
                # to know which point was clicked. We do an extra render pass
                # of the region around the mouseclick, with each point
                # rendered in a unique color and without blending and
                # antialiasing.
                pos = canvas.transforms.canvas_transform.map(event.pos)
                try:
                    p1.update_gl_state(blend=False)
                    p1.antialias = 0
                    p1.set_data(positions, face_color=ids, **kwargs)
                    img = canvas.render((pos[0] - click_radius,
                                         pos[1] - click_radius,
                                         click_radius * 2 + 1,
                                         click_radius * 2 + 1),
                                        bgcolor=(0, 0, 0, 0))
                finally:
                    p1.update_gl_state(blend=True)
                    p1.antialias = 1
                    p1.set_data(positions, face_color=colors, **kwargs)
                # We pick the pixel directly under the click, unless it is
                # zero, in which case we look for the most common nonzero
                # pixel value in a square region centered on the click.
                idxs = img.ravel().view(np.uint32)
                idx = idxs[len(idxs) // 2]
                if idx == 0:
                    idxs, counts = np.unique(idxs, return_counts=True)
                    idxs = idxs[np.argsort(counts)]
                    idx = idxs[-1] or (len(idxs) > 1 and idxs[-2])
                # call the callback function
                if idx > 0:
                    # subtract one; color 0 was reserved for the background
                    on_click(idx - 1, p1)

        canvas.events.mouse_release.connect(on_mouse_release)

    # run application
    app.run()


if __name__ == "__main__":
    data = np.random.randn(1000, 3)
    cmap = color.get_colormap('viridis')
    colors = np.concatenate(list(map(cmap.map, np.linspace(0, 1, len(data)))))
    def on_click(idx, markers):
        # turn the clicked marker white just for demonstration
        colors[idx] = (1, 1, 1, 1)
        markers.set_data(data, face_color=colors, symbol='o', size=4.5,
                         edge_color=None)

    scatter3d(data, colors, on_click=on_click)