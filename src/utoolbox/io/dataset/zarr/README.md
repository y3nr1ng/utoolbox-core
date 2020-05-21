# Zarr Dataset

Zarr provides an implementation of chunked, compressed, N-dimensional arrays. In order to accomodate our common acquisition requirements, it utilize meta-data and a hierarchy group similar to the container structure used in [BDV](https://arxiv.org/pdf/1412.0488.pdf), [N5](https://github.com/saalfeldlab/n5-viewer), and [IMS](http://open.bitplane.com/Default.aspx?tabid=268).

## Container Structure
A dataset stores not only the original image data, but also other lower resolution and post-process generated data.

### General container structure
Starting from the predefined root group, hierarchy structure of the dataset includes
```
/t{:d}/c{:d}/s{:d}/{:s}/{:d}
```
each level corresponds to 
- `t` frame
- `c` channel
- `s` spatial setup, includes different views and tiles
- last level contains either a simple or multiscale array
    - simple, an array
    - multiscale, which is in fact, a group contains multiple array

### Dataset root
For the original dataset, it is stored under `/`.

### Archive
To archive a dataset, it is recommended to keep only zero resolution level of the raw array.

If desired, one may keep preview related arrays to provide quick preview access.

## Attributes
### Signature
Root level of the Zarr file should contain
- `zarr_dataset` magic attribute with value `ZarrDataset`
- `format_version` implies the version that implements the container format

### TODO
TODO
- define time info, channel info, spatial setup info (coordinate, resolution, downsample setup)