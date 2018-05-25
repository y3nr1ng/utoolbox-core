import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt

def orthoview(volume, cmap='gray', margin=0.05, dpi=80):
    if volume.ndim != 3:
        raise RuntimeError("image should be 3D")

    try:
        spacing = volume.metadata.spacing
    except AttributeError:
        spacing = (1, ) * volume.ndim

    nz, ny, nx = volume.shape
    figsize = ((1+margin) * ny/dpi, (1+margin) * nx/dpi)
    extent = (0, ny*spacing[1], nx*spacing[2], 0)

    # create named display
    handle = display(None, display_id=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1-2*margin, 1-2*margin])
    plt.set_cmap(cmap)
    im = ax.imshow(volume[nz//2, ...], extent=extent, interpolation=None)

    slider = widgets.IntSlider(min=0, max=nz-1, value=nz//2)
    def slide_changed(z):
        print("z.new={}".format(z.new))
        im.set_array(volume[z.new, ...])
        handle.update(fig)
        fig.canvas.draw()

    slider.observe(slide_changed, names='value')
    display(slider)
