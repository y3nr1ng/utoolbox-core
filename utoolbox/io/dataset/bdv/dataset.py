from functools import partial, reduce
import logging
import operator
import os
from xml.etree.ElementTree import Element, ElementTree, SubElement

from dask.distributed import as_completed
import h5py
import numpy as np

from ..base import DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset

__all__ = ["BigDataViewerDataset"]

logger = logging.getLogger(__name__)


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

        print(path)
        print(self._path)
        print(self.path)

        nslots = reduce(operator.mul, cache_n_chunks, 1)
        nbytes = reduce(operator.mul, cache_chunk_size, 1) * nslots * 2
        logger.info(f"cache size: {nbytes} bytes")

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

    def close(self):
        self.handle.close()
        self._handle = None

    def open(self):
        self._handle = self._func()

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

        # generate VDS filename
        fname, fext = os.path.splitext(self.path)
        fname = f"{fname}_{ss:02d}{fext}"

        tasks = []
        for i, (bins, chunk) in enumerate(zip(pyramid, chunks)):
            # new dataset
            path = f"t{0:05d}/s{ss:02d}/{i}"
            logger.debug(f".. > {path}, bin:{bins}, chunk:{chunk}")

            # downsample range
            ranges = tuple(slice(None, None, step) for step in bins)
            sdata = data[ranges]

            print(f"{sdata.shape}, {sdata.dtype}")

            layout = h5py.VirtualLayout(shape=sdata.shape, dtype=sdata.dtype)
            layout[:] = h5py.VirtualSource(
                fname, f"{i}", shape=sdata.shape, dtype=sdata.dtype
            )

            # data and assignments
            tasks.append((f"{i}", sdata))

            group = self.handle.create_group(path)
            group.create_virtual_dataset("cells", layout)

            # group.create_dataset(
            #    "cells",
            #    data=sdata,
            #    chunks=chunk,
            #    scaleoffset=0,
            #    compression=compression,
            #    shuffle=True,
            # )

            ## Assemble virtual dataset
            # layout = h5py.VirtualLayout(shape=(4, 100), dtype="i4")
            # for n in range(4):
            #    filename = "{}.h5".format(n)
            #    vsource = h5py.VirtualSource(filename, "data", shape=(100,))
            #    layout[n] = vsource

            ## Add virtual dataset to output file
            # with h5py.File("VDS.h5", "w", libver="latest") as f:
            #    f.create_virtual_dataset("vdata", layout, fillvalue=-5)

        self.handle.flush()

        return fname, tasks

    ##

    @classmethod
    def _write_matrix(cls, handle, name, matrix, can_append=True):
        matrix = np.array(matrix)
        matrix = np.fliplr(matrix)  # HDF5 use different order
        if can_append:
            shape = (None,) + matrix.shape[1:]
        handle.create_dataset(name, data=matrix, maxshape=shape)


class BigDataViewerDataset(
    DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset
):
    def __init__(self, root_dir):
        self._root_dir = root_dir

        super().__init__()

        self.preload()

    ##

    @property
    def read_func(self):
        pass

    @property
    def root_dir(self):
        return self._root_dir

    ##

    @staticmethod
    def dump(
        dst_dir,
        dataset,
        pyramid=[(1, 1, 1), (2, 4, 4)],
        chunks=(64, 128, 128),
        compression="gzip",
        client=None,
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
        voxel_size = dataset._load_voxel_size()

        shape, _ = dataset._load_array_info()

        def _serialize_h5(fname, level, data):
            print(f".. {fname}, {level}, LOAD")
            data = data.compute()
            with h5py.File(fname, "a") as h:
                print(f".. {fname}, {level}, SAVE")
                h.create_dataset(level, data=data)
            print(f".. {fname}, {level}, DELETE")
            del data

            return fname

        futures = []
        with BigDataViewerHDF5(h5_path, "w") as h:
            for coords, uuid in dataset.inventory.items():
                coord_dict = {
                    k: v for k, v in zip(dataset.inventory.index.names, coords)
                }

                # generate queries
                statements = [
                    f"{k}=={coord_dict[k]}" for k in ("tile_x", "tile_y", "tile_z")
                ]
                query_stmt = " & ".join(statements)
                # find tile linear index
                index = dataset.tile_coords.query(query_stmt).index.values
                index = index[0]

                ss = xml.add_view(
                    shape,
                    name=uuid,
                    voxel_size=voxel_size,
                    channel=coord_dict["channel"],
                    illumination=coord_dict["view"],
                    tile=index,
                )
                logger.info(f" [{ss}] {uuid}")

                # transformation
                matrix = np.zeros((3, 4))
                matrix[range(3), range(3)] = 1
                matrix[range(3), -1] = [
                    coord_dict[k] / f * s
                    for k, f, s in zip(
                        ("tile_y", "tile_z", "tile_x"),
                        reversed(voxel_size),
                        (-1, -1, -1),
                    )
                ]
                xml.views[ss].add_transform("Translation to Regular Grid", matrix)

                # write data
                fname, levels = h.add_view(ss, dataset[uuid], pyramid, chunks, "gzip")
                for name, data in levels:
                    future = client.submit(_serialize_h5, fname, name, data)
                    futures.append(future)

        xml.serialize()

        logger.info(f"{len(futures)} file(s) to generate")
        for i, (future, result) in enumerate(as_completed(futures, with_results=True)):
            logger.debug(f".. {i+1}/{len(futures)}, {result}")

    ##

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

    def _load_tiling_coordinates(self):
        pass

    def _load_tiling_info(self):
        pass

    def _load_view_info(self):
        pass
