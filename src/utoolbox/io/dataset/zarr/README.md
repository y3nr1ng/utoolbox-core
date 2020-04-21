# Zarr Dataset

Zarr provides an implementation of chunked, compressed, N-dimensional arrays. In order to accomodate our common acquisition requirements, it utilize meta-data and a hierarchy group similar to the container structure used in [BDV](https://arxiv.org/pdf/1412.0488.pdf), [N5](https://github.com/saalfeldlab/n5-viewer), and [IMS](http://open.bitplane.com/Default.aspx?tabid=268).

## Container Structure
A dataset stores not only the original image data, but also other lower resolution and post-process generated data.

### General container structure
Starting from the predefined root group, hierarchy structure of the dataset includes
```
/t{:d}/c{:d}/s{:d}/{:d}
```
each level corresponds to 
- `t` frame
- `c` channel
- `s` spatial setup, includes different views and tiles
- last level defines the pyramid scale level, starts from `0`, the original resolution

### Attributes
TODO
- define time info, channel info, spatial setup info (coordinate, resolution, downsample setup)

### Dataset root
For the original dataset, it is stored under `/`, where generated dataset has an underscore prefix, `/_{:s}`, as its root.

### Archive
To archive a dataset, it is recommended to remove root groups start with an underscore, and all the non-zero resolution level.

If desired, one may keep the `/_preview` root group to provide quick preview access.
