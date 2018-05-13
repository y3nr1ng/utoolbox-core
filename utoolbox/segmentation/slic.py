import numpy as np
from skimage.util import img_as_float, regular_grid

try:
    from ._slic import slic_cython
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
    from ._slic import slic_cython

def slic(image, n_segments=None, compactness=.1, max_iter=10, spacing=None):
    """
    Segments image using k-means clustering in cartesian space.

    Parameters
    ----------
    image : 2D or 3D ndarray
        Input image, which can be 2D or 3D grayscale image.
    n_segments : int, optional
        The approximate number of labels in the segmented output image.
    compactness : float, optional
        Balance space proximity. Higher value gives more initial weight to space
        proximity, making superpixel shapes more square/cubic. This parameter
        depends strongly on image contrast and on the shapes of objects in the
        image.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    spacing : (2, ) or (3, ) array-like of floats, optional
        The voxel spacing along each image dimension. By default, image is
        assumed to be uniform spaced. This parameter controls the weights of the
        distances along each dimension during k-means clustering.

    Returns
    -------
    labels : 2D or 3D ndarray
        Integer mask indicating segment labels.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
    .. [2] http://ivrg.epfl.ch/research/superpixels#SLICO
    .. [3] http://scikit-image.org/docs/dev/api/skimage.segmentation.html
    """
    if image.ndim != 2 and image.ndim != 3:
        raise ValueError("Invalid image dimension ({}).".format(image.ndim))
    image = img_as_float(image)

    #TODO adapt for 2D array, for now, assume it is targeted for 3D
    if spacing is None:
        spacing = np.ones(3)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.asarray(spacing, dtype=np.double)

    depth, height, width = image.shape

    # initialize cluster centroids for desired number of segments
    grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
    slices = regular_grid(image.shape[:3], n_segments)
    step_z, step_y, step_x = [int(s.step if s.step is not None else 1)
                              for s in slices]

    # centroid coordinate
    segments_z = grid_z[slices]
    segments_y = grid_y[slices]
    segments_x = grid_x[slices]
    # segment no.
    segments_c = np.zeros_like(segments_z, dtype=np.double)

    segments = np.stack((segments_z, segments_y, segments_x, segments_c),
                        axis=3).reshape(-1, 4)
    segments = np.ascontiguousarray(segments)

    # scaling of the ratio
    step = float(max((step_z, step_y, step_x)))
    ratio = 1. / compactness
    image = np.ascontiguousarray(image*ratio)

    print(segments.shape)
    print(segments.dtype)

    labels = slic_cython(image, segments, step, max_iter, spacing)
    return labels
