import logging
import operator
import os
from functools import partial, reduce
from xml.etree.ElementTree import Element, ElementTree, SubElement

import h5py
import numpy as np
from humanfriendly import format_size

from ..base import (
    TILE_INDEX_STR,
    DenseDataset,
    MultiChannelDataset,
    MultiChannelDatasetIterator,
    MultiViewDataset,
    MultiViewDatasetIterator,
    SessionDataset,
    TiledDataset,
    TiledDatasetIterator,
)
from .error import InvalidChunkSizeError

__all__ = ["BigDataViewerDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class BigDataViewerXML(object):
    """
    Args:
        path (str): path to the dataset
    """

    class View(object):
        attributes = dict()

        def __init__(
            self, vid, shape, name="untitled", voxel_size=(1, 1, 1), **attributes
        ):
            self.vid = vid
            self.shape = shape
            self.name = name
            self.voxel_size = voxel_size
            self.attributes = {
                key: BigDataViewerXML.View.archive_attribute(key, value)
                for key, value in attributes.items()
            }

            self.reset_transform()

        def add_transform(self, name, matrix):
            self.transforms.append((name, matrix))

        def reset_transform(self):
            self.transforms = []

            voxel_size = self.voxel_size[::-1]
            # upsample low-res axes
            min_voxel_size = min(voxel_size)
            voxel_size = tuple(s / min_voxel_size for s in voxel_size)

            matrix = np.zeros((3, 4))
            matrix[range(3), range(3)] = voxel_size
            self.add_transform("calibration", matrix)

        def serialize(self):
            # abstract definitions
            setup = Element("ViewSetup")
            SubElement(setup, "id").text = str(self.vid)
            SubElement(setup, "name").text = self.name
            SubElement(setup, "size").text = " ".join(
                str(s) for s in reversed(self.shape)
            )

            # attach attributes
            attributes = SubElement(setup, "attributes")
            for key, index in self.attributes.items():
                SubElement(attributes, key).text = str(index)

            # spatial calibrations
            voxel = SubElement(setup, "voxelSize")
            SubElement(voxel, "unit").text = "micron"
            SubElement(voxel, "size").text = " ".join(
                str(s) for s in reversed(self.voxel_size)
            )

            transforms = Element("ViewRegistration")
            transforms.set("timepoint", str(0))
            transforms.set("setup", str(self.vid))
            for name, matrix in reversed(self.transforms):
                transform = SubElement(transforms, "ViewTransform")
                transform.set("type", "affine")
                SubElement(transform, "Name").text = name
                SubElement(transform, "affine").text = " ".join(
                    "{:.4f}".format(v) for v in matrix.ravel()
                )

            return setup, transforms

        ##

        @classmethod
        def archive_attribute(cls, key, value):
            """
            Archive an attribute and returns its respective ID.
            """
            # XML can only accept strings
            value = str(value)
            try:
                attribute = cls.attributes[key]
                try:
                    return attribute.index(value)
                except ValueError:
                    # new value
                    attribute.append(value)
                    return len(attribute) - 1
            except KeyError:
                # new attribute
                cls.attributes[key] = [value]
                return 0

    ##

    def __init__(self, path):
        path = os.path.realpath(path)
        self._init_tree(path)

        # XML will place next to the dataset
        fname, _ = os.path.splitext(path)
        self._path = f"{fname}.xml"

        self._views = []

    ##

    @property
    def path(self):
        return self._path

    @property
    def root(self):
        return self._root

    @property
    def views(self):
        return self._views

    ##

    def add_view(self, data, name="untitled", voxel_size=(1, 1, 1), **kwargs):
        """
        Add a new view and return its stored view ID.
        """
        vid = len(self._views)
        if "tile" not in kwargs:
            kwargs["tile"] = vid
        view = BigDataViewerXML.View(
            vid, data, name=name, voxel_size=voxel_size, **kwargs
        )
        self._views.append(view)
        return vid

    def serialize(self):
        for view in self._views:
            setup, transforms = view.serialize()
            self._setups.append(setup)
            self._registrations.append(transforms)

        for key, values in BigDataViewerXML.View.attributes.items():
            attribute = SubElement(self._setups, "Attributes")
            attribute.set("name", key)
            for i, value in enumerate(values):
                variants = SubElement(attribute, key.capitalize())
                SubElement(variants, "id").text = str(i)
                SubElement(variants, "name").text = str(value)

        tree = ElementTree(self.root)
        tree.write(self.path)
        logger.info(f'XML saved to "{self.path}"')

    ##

    def _init_tree(self, h5_path):
        # init XML
        root = Element("SpimData")
        root.set("version", "0.2")

        # using relative path
        base_path = SubElement(root, "BasePath")
        base_path.set("type", "relative")
        base_path.text = "."

        sequence = SubElement(root, "SequenceDescription")

        # a HDF data source
        loader = SubElement(sequence, "ImageLoader")
        loader.set("format", "bdv.hdf5")
        loader_path = SubElement(loader, "hdf5")
        loader_path.set("type", "relative")
        loader_path.text = os.path.basename(h5_path)

        # populate default fields
        setups = SubElement(sequence, "ViewSetups")
        timepoints = SubElement(sequence, "Timepoints")
        timepoints.set("type", "pattern")
        registrations = SubElement(root, "ViewRegistrations")

        # TODO no timeseries
        SubElement(timepoints, "integerpattern").text = str(0)

        # save internal arguments
        self._root = root
        self._setups, self._registrations = setups, registrations


class BigDataViewerHDF5(object):
    def __init__(
        self, path, mode, cache_chunk_size=(64, 64, 64), cache_n_chunks=(1, 4, 4)
    ):
        self._path = path

        nslots = reduce(operator.mul, cache_n_chunks, 1)
        nbytes = reduce(operator.mul, cache_chunk_size, 1) * nslots * 2
        logger.info(f"cache size: {format_size(nbytes, binary=True)}")

        self._func = partial(
            h5py.File, self.path, mode, rdcc_nbytes=nbytes, rdcc_nslots=nslots
        )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    ##

    @property
    def handle(self):
        return self._handle

    @property
    def path(self):
        return self._path

    ##

    def add_view(self, ss, data, pyramid, chunks, compression):
        sname = f"s{ss:02d}"
        info = self.handle.create_group(sname)
        self._write_matrix(info, "resolutions", pyramid)
        if isinstance(chunks[0], int):
            # reuse same chunk size for each level
            chunks = (chunks,)
            chunks *= len(pyramid)
        self._write_matrix(info, "subdivisions", chunks)

        # NOTE BDV reads int16 instead of uint16 internally
        data = data.astype(np.int16)

        for i, (bins, chunk) in enumerate(zip(pyramid, chunks)):
            # new dataset
            path = f"t{0:05d}/s{ss:02d}/{i}"
            logger.debug(f".. > {path}, bin:{bins}, chunk:{chunk}")

            # downsample range
            ranges = tuple(slice(None, None, step) for step in bins)
            sdata = data[ranges]

            # kick start
            sdata = sdata.compute()

            group = self.handle.create_group(path)
            try:
                group.create_dataset(
                    "cells", data=sdata, chunks=chunk, compression=compression
                )
            except ValueError as err:
                err_str = str(err)
                raise InvalidChunkSizeError(err_str)
            finally:
                self.handle.flush()

    def close(self):
        self.handle.close()
        self._handle = None

    def open(self):
        self._handle = self._func()

    ##

    @classmethod
    def _write_matrix(cls, handle, name, matrix, can_append=True):
        matrix = np.array(matrix)
        matrix = np.fliplr(matrix)  # HDF5 use different order
        if can_append:
            shape = (None,) + matrix.shape[1:]
        handle.create_dataset(name, data=matrix, maxshape=shape)


class DummyBigDataViewerHDF5(BigDataViewerHDF5):
    def add_view(self, *args, **kwargs):
        pass

    def open(self):
        logger.info("BigDataViewerDataset DRY RUN dump()")

    def close(self):
        pass


class BigDataViewerDataset(
    SessionDataset, DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset
):
    """
    Using HDF5-backend BigDataViewer format.

    Args:
        TBD
    """

    def __init__(self, root_dir: str):
        super().__init__(store=root_dir, path="/")  # BDV does not allow arbitrary root

        self._xml_handle, self._h5_handle = None, None

    ##

    @property
    def h5(self) -> BigDataViewerHDF5:
        return self._h5_handle

    @property
    def read_func(self):
        pass

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def xml(self) -> BigDataViewerXML:
        return self._xml_handle

    ##

    @classmethod
    def dump(
        cls,
        dst_dir: str,
        dataset,
        pyramid=[(1, 1, 1), (2, 4, 4)],
        chunks=(64, 128, 128),
        compression="gzip",
        dry_run=False,
    ):
        try:
            os.makedirs(dst_dir)
        except FileExistsError:
            logger.warning(f'folder "{dst_dir}" already exists')
        h5_path = os.path.join(dst_dir, f"dataset.h5")

        xml = BigDataViewerXML(h5_path)
        if not isinstance(dataset, DenseDataset):
            raise TypeError("dataset is not a DenseDataset")
        voxel_size = dataset.voxel_size

        shape, _ = dataset._load_array_info()

        if dry_run:
            klass = DummyBigDataViewerHDF5
        else:
            klass = BigDataViewerHDF5

        with klass(h5_path, "w") as h:
            for channel, c_selected in MultiChannelDatasetIterator(dataset):
                for view, v_selected in MultiViewDatasetIterator(c_selected):
                    kwargs = {"channel": channel, "illumination": view}

                    for (index, coord), selected in TiledDatasetIterator(
                        v_selected, axes="xyz", return_key=True, return_format="both"
                    ):
                        # find tile linear index
                        coords = {
                            k: v for k, v in zip(sorted(TILE_INDEX_STR), index)
                        }  # in XYZ order
                        result = dataset.tile_coords.xs(
                            list(coords.values()),
                            axis="index",
                            level=list(coords.keys()),
                        )
                        row_index = result.iloc[0].name
                        linear_index = dataset.tile_coords.index.get_loc(row_index)

                        uuid = selected.inventory.values[0]

                        ss = xml.add_view(
                            shape,
                            name=uuid,
                            voxel_size=voxel_size,
                            tile=linear_index,
                            **kwargs,
                        )
                        logger.info(f" [{ss}] {uuid}")

                        # anisotropic factor
                        min_voxel_size = min(voxel_size)
                        factor = tuple(s / min_voxel_size for s in voxel_size)

                        # anisotropic factor
                        min_voxel_size = min(voxel_size)
                        factor = tuple(s / min_voxel_size for s in voxel_size)

                        # 3d transformation, pad info if missing
                        if len(coord) == 2:
                            coord = coord + (0,)

                        # transformation
                        matrix = np.zeros((3, 4))
                        matrix[range(3), range(3)] = 1
                        matrix[range(3), -1] = [
                            c / v * s * f
                            for c, v, s, f in zip(
                                coord,
                                reversed(voxel_size),
                                (-1, -1, -1),
                                reversed(factor),
                            )
                        ]
                        xml.views[ss].add_transform(
                            "Translation to Regular Grid", matrix
                        )

                        # write data
                        h.add_view(ss, dataset[selected], pyramid, chunks, compression)

            xml.serialize()

    ##

    def _open_session(self):
        # TODO use HDF5/XML instance object instead of handle
        pass

    def _close_session(self):
        pass

    def _can_read(self):
        pass

    def _enumerate_files(self):
        pass

    def _load_array_info(self):
        pass

    def _load_channel_info(self):
        pass

    def _retrieve_file_list(self, coord_dict):
        pass

    def _load_metadata(self):
        pass

    def _load_coordinates(self):
        pass

    def _load_tiling_info(self):
        pass

    def _load_view_info(self):
        pass
