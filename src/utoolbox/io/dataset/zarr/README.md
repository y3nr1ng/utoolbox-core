# Zarr Dataset

Zarr provides an implementation of chunked, compressed, N-dimensional arrays. In order to accomodate our common acquisition requirements, it utilize meta-data and a hierarchy group similar to the container structure used in BDV, N5, and IMS.

## Container Structure
A dataset stores not only the original image data, but also allowed lower resolution and other post-process generated data.

Raw dataset
'''
/t{:d}/c{:d}/s{:d}/{:d}
'''

Generated dataset
'''
/_{:s}/...
'''

