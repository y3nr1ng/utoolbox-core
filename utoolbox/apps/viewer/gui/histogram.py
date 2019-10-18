import numpy as np

from vispy.ext.six import string_types
from vispy.visuals.mesh import MeshVisual
from vispy.scene.visuals import create_visual_node

__all__ = ["Histogram", "HistogramVisual"]


class HistogramVisual(MeshVisual):
    """
    Modified visual that only displays histogram.

    Args:
        histogram (array-like): histogram data, follow numpy format
        color (optional): color of the histogram instance
        orientation (str, optional): orientation of the histogram, 'h' or 'v'
    """

    def __init__(self, histogram, color="w", orientation="h"):
        #   4-5
        #   | |
        # 1-2/7-8
        # |/| | |
        # 0-3-6-9
        if not isinstance(orientation, string_types) or orientation not in ("h", "v"):
            raise ValueError('orientation must be "h" or "v", not %s' % (orientation,))
        X, Y = (0, 1) if orientation == "h" else (1, 0)

        # do the histogramming
        data, bin_edges = histogram
        # construct our vertices
        rr = np.zeros((3 * len(bin_edges) - 2, 3), np.float32)
        rr[:, X] = np.repeat(bin_edges, 3)[1:-1]
        rr[1::3, Y] = data
        rr[2::3, Y] = data
        bin_edges.astype(np.float32)
        # and now our tris
        tris = np.zeros((2 * len(bin_edges) - 2, 3), np.uint32)
        offsets = 3 * np.arange(len(bin_edges) - 1, dtype=np.uint32)[:, np.newaxis]
        tri_1 = np.array([0, 2, 1])
        tri_2 = np.array([2, 0, 3])
        tris[::2] = tri_1 + offsets
        tris[1::2] = tri_2 + offsets
        MeshVisual.__init__(self, rr, tris, color=color)


Histogram = create_visual_node(HistogramVisual)
