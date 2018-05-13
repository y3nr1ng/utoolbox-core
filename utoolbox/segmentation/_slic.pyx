#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX
from libc.stdio cimport fprintf, stderr

import numpy as np

from skimage.util import regular_grid

def slic_cython(double[:, :, ::1] image_zyx,
                double[:, ::1] segments,
                float step,
                Py_ssize_t max_iter,
                double[::1] spacing):
    """Helper function for SLIC segmentation.

    Parameters
    ----------
    image_zyx : 3D array of float, shape (Z, Y, X)
        The input image.
    segments : 2D array of float, shape (N, 3)
        The initial centroids obtained by SLIC as [Z, Y, X].
    max_iter : int
        The maximum number of k-means iterations.
    spacing : 1D array of float, shape (3, )
        The voxel spacing along each image dimension. By default, image is
        assumed to be uniform spaced. This parameter controls the weights of the
        distances along each dimension during k-means clustering.

    Returns
    -------
    nearest_segments : 3D array of int, shape (Z, Y, X)
        The label superpixels found by SLIC.
    """
    # initialize on grid
    cdef Py_ssize_t depth, height, width
    depth = image_zyx.shape[0]
    height = image_zyx.shape[1]
    width = image_zyx.shape[2]

    cdef Py_ssize_t n_segments = segments.shape[0]
    # number of features [Z, Y, X]
    cdef Py_ssize_t n_features = segments.shape[1]

    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    slices = regular_grid((depth, height, width), n_segments)
    step_z, step_y, step_x = [int(s.step if s.step is not None else 1)
                              for s in slices]

    cdef Py_ssize_t[:, :, ::1] nearest_segments = \
        np.empty((depth, height, width), dtype=np.intp)
    cdef double[:, :, ::1] distance = \
        np.empty((depth, height, width), dtype=np.double)
    cdef Py_ssize_t[::1] n_segment_elems = np.zeros(n_segments, dtype=np.intp)

    cdef Py_ssize_t i, c, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max
    cdef char change
    cdef double dist_center, cx, cy, cz, dy, dz

    cdef double sz, sy, sx
    sz = spacing[0]
    sy = spacing[1]
    sx = spacing[2]

    # The colors are scaled before being passed to _slic_cython so
    # max_color_sq can be initialised as all ones
    cdef double[::1] max_dist_color = np.ones(n_segments, dtype=np.double)
    cdef double dist_color

    # The reference implementation (Achanta et al.) calls this invxywt
    cdef double spatial_weight = 1. / (step ** 2)

    with nogil:
        for i in range(max_iter):
            fprintf(stderr, "iter=%ld\n", i)

            change = 0
            distance[:, :, :] = DBL_MAX

            # assign pixels to segments
            for k in range(n_segments):
                # segment coordinate centers
                cz = segments[k, 0]
                cy = segments[k, 1]
                cx = segments[k, 2]

                # compute windows
                z_min = <Py_ssize_t>max(cz - 2 * step_z, 0)
                z_max = <Py_ssize_t>min(cz + 2 * step_z + 1, depth)
                y_min = <Py_ssize_t>max(cy - 2 * step_y, 0)
                y_max = <Py_ssize_t>min(cy + 2 * step_y + 1, height)
                x_min = <Py_ssize_t>max(cx - 2 * step_x, 0)
                x_max = <Py_ssize_t>min(cx + 2 * step_x + 1, width)

                for z in range(z_min, z_max):
                    dz = (sz * (cz - z)) ** 2
                    for y in range(y_min, y_max):
                        dy = (sy * (cy - y)) ** 2
                        for x in range(x_min, x_max):
                            dist_center = (dz + dy + (sx * (cx - x)) ** 2) * spatial_weight
                            dist_color = (image_zyx[z, y, x] - segments[k, 3]) ** 2

                            dist_center += distn_color / max_dist_color[k]

                            if distance[z, y, x] > dist_center:
                                nearest_segments[z, y, x] = k
                                distance[z, y, x] = dist_center
                                change = 1

            # stop if no pixel changed its segment
            if change == 0:
                break

            # recompute segment centers

            # sum features for all segments
            n_segment_elems[:] = 0
            segments[:, :] = 0
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        k = nearest_segments[z, y, x]
                        n_segment_elems[k] += 1
                        segments[k, 0] += z
                        segments[k, 1] += y
                        segments[k, 2] += x

                        segments[k, 3] = image_zyx[z, y, x]

            # divide by number of elements per segment to obtain mean
            for k in range(n_segments):
                for c in range(n_features):
                    segments[k, c] /= n_segment_elems[k]

            # update the color distance maxima
            for z in range(depth):
                for y in range(height):
                    for x in range(width):

                        k = nearest_segments[z, y, x]
                        dist_color = (image_zyx[z, y, x] - segments[k, 3]) ** 2

                        # The reference implementation seems to only change
                        # the color if it increases from previous iteration
                        if max_dist_color[k] < dist_color:
                            max_dist_color[k] = dist_color

    return np.asarray(nearest_segments)
