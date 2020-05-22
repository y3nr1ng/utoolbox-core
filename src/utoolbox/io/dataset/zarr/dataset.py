import logging
import os
from collections import defaultdict
from typing import List, Optional

import dask.array as da
import numpy as np
import pandas as pd
import xxhash
import zarr
from dask import delayed
from dask.distributed import as_completed

from utoolbox.utils.dask import get_client, wait_futures

from ..base import (
    DenseDataset,
    MultiChannelDataset,
    MultiChannelDatasetIterator,
    MultiViewDataset,
    MultiViewDatasetIterator,
    SessionDataset,
    TiledDataset,
    TiledDatasetIterator,
    TimeSeriesDataset,
    TimeSeriesDatasetIterator,
)

__all__ = ["ZarrDataset"]

logger = logging.getLogger("utoolbox.io.dataset")

RAW_DATA_LABEL = "raw"


class ZarrDataset(
    SessionDataset,
    DenseDataset,
    MultiChannelDataset,
    MultiViewDataset,
    TiledDataset,
    TimeSeriesDataset,
):
    version = 1

    """
    Using Zarr directory as backend. 

    Default internal path is '/', post-processing steps use groups with '_' 
    prefix as their root.

    Args:
        store (str): path to the data store
        label (str, optional): label of the data array
        level (int, optional): resolution level
        path (str, optional): internal group path
    """

    def __init__(
        self, store: str, label: str = RAW_DATA_LABEL, level: int = 0, path: str = "/",
    ):
        if level < 0:
            raise ValueError("resolution level should >= 0")
        self._label, self._level = label, level

        super().__init__(store, path)

    ##

    @property
    def label(self) -> str:
        return self._label

    @property
    def level(self) -> int:
        """Pyramid level to load."""
        return self._level

    @level.setter
    def level(self, level: int):
        # TODO how to reload dataset from filesystem
        raise NotImplementedError

    @property
    def read_func(self):
        def func(uri, shape, dtype):
            # zarr array contains shape, dtype and decompression info
            return da.from_zarr(self.root_dir, uri)

        return func

    ##

    @classmethod
    def dump(
        cls,
        store: str,
        dataset,
        label: str = RAW_DATA_LABEL,
        path: Optional[str] = None,
        overwrite=False,
        client=None,
        **kwargs,
    ):
        """
        Dump dataset into Zarr dataset.

        The target storing location is defined by:
        - label, name of this dataset
        - level, dump operation always assume it has the highest resolution
        - path, what internal location (generally, at root, '/')

        Args:
            store (str): path to the data store
            dataset : serialize the provided dataset
            label (str, optional): label of the data array
            path (str, optional): internal group path
            overwrite (bool, optional): overwrite _data_ if it already exists
            client (Client, optional): remote cluster client
            **kwargs : additional argument for `zarr.open` function

        Returns:
            (list of Futures)

        Note:
            Even if overwrite is False, existing attributes _WILL BE_ overwrite.
        """
        kwargs["mode"] = "a"
        root = zarr.open(store, **kwargs)

        # test attributes
        root.attrs["zarr_dataset"] = "ZarrDataset"
        root.attrs["format_version"] = cls.version

        # TODO save original metadata, msgpack?

        if path:
            root = root.open_group(path, mode="w")

        hash_gen = xxhash.xxh64()

        client = client if client else get_client()
        futures = {}  # conversion tasks, batch submit after populated all of them

        # start populating the container structure
        #   /time/channel/setup/level
        # welp, i have no idea how to do this cleanly without nested structure
        # 1) time
        for i_t, (t, t_selected) in enumerate(TimeSeriesDatasetIterator(dataset)):
            t_root = root.require_group(f"t{i_t}")

            if t is None:
                try:
                    del t_root.attrs["timestamp"]
                except KeyError:
                    pass
            else:
                # convert from np.datetime64 to int (JSON serializable)
                t = t.to_timedelta64()  # ns
                t = int(t) // 1000000  # ms
                t_root.attrs["timestamp"] = t

            # 2) channel
            for i_c, (c, c_selected) in enumerate(
                MultiChannelDatasetIterator(t_selected)
            ):
                c_root = t_root.require_group(f"c{i_c}")
                c_root.attrs["channel"] = c  # NOTE required

                # 3) setup
                i_s = 0
                for sv, v_selected in MultiViewDatasetIterator(c_selected):
                    tiles_iterator = TiledDatasetIterator(
                        v_selected, axes="zyx", return_format="both"
                    )
                    for st, selected in tiles_iterator:
                        s_root = c_root.require_group(f"s{i_s}")
                        i_s += 1

                        # view attribute
                        if sv is None:
                            try:
                                del s_root.attrs["view"]
                            except KeyError:
                                pass
                        else:
                            s_root.attrs["view"] = sv

                        # tile attribute
                        if st is None:
                            for index in ("tile_index", "tile_coord"):
                                try:
                                    del s_root.attrs[index]
                                except KeyError:
                                    pass
                        else:
                            index, coord = st
                            names = tiles_iterator.index
                            names = [name.split("_")[-1] for name in names]
                            # NOTE numpy dtypes are not serializable, use native
                            s_root.attrs["tile_index"] = {
                                k: int(v) for k, v in zip(names, index)
                            }
                            s_root.attrs["tile_coord"] = {
                                k: float(v) for k, v in zip(names, coord)
                            }

                        # 4-1) retrieve info
                        data = dataset[selected]

                        # 4-2) label
                        write_back = True
                        if label in s_root:
                            label_group = s_root[label]

                            # get the hash
                            hash_gen.update(data.compute())
                            src_hash = hash_gen.hexdigest()
                            hash_gen.reset()

                            try:
                                dst_hash = label_group.attrs["checksum"]
                                if dst_hash != src_hash:
                                    raise ValueError
                            except KeyError:
                                # hash does not exist, since it is only updated when
                                # dump was completed
                                logger.warning(
                                    f'"{label_group.path}" contains partial dump, rewrite'
                                )
                            except ValueError:
                                # hash mismatch
                                logger.warning(
                                    "existing hash does not match the source"
                                )
                                # reset
                                del label_group.attrs["checksum"]
                            else:
                                # array exists, and checksum matches
                                if not overwrite:
                                    logger.info(f'"{label_group.path}" already exists')
                                    write_back = False
                        else:
                            # never seen this label before, create new one
                            # NOTE raw data will always be multiscale
                            label_group = s_root.require_group(label)
                            label_group.attrs["voxel_size"] = dataset.voxel_size

                        # 5) level
                        # NOTE comply with multiscale arrays v0.1
                        # https://forum.image.sc/t/multiscale-arrays-v0-1/37930

                        # drop current multiscale attributes
                        if "multiscales" in label_group.attrs:
                            # delete all existing levels
                            multiscales = label_group.attrs["multiscales"]
                            if multiscales:
                                logger.warning("deleting existing multiscale datasets")
                                for multiscale in multiscales:
                                    for path in multiscale["datasets"]:
                                        try:
                                            del label_group[path["path"]]
                                        except KeyError:
                                            logger.warning(
                                                f'"{path}" is already deleted'
                                            )

                        # generate 0-level, single-level only
                        level_str = str("0")
                        multiscales = [
                            {
                                "name": label,
                                "datasets": [{"path": level_str}],
                                "version": "0.1",
                            }
                        ]
                        if level_str in label_group:
                            data_dst = label_group[level_str]
                        else:
                            # NOTE compression benchmark reference http://alimanfoo.
                            # github.io/2016/09/21/genotype-compression-benchmark.html
                            data_dst = label_group.empty_like(
                                level_str,
                                data,
                                chunks=True,
                                compression="blosc",
                                compression_opts=dict(
                                    cname="lz4", clevel=5, shuffle=zarr.blosc.SHUFFLE
                                ),
                            )
                        label_group.attrs["multiscales"] = multiscales

                        # complete updates all the attributes, early stop here
                        if not write_back:
                            continue

                        # NOTE using default chunk shape after rechunk will cause
                        # problem, since chunk size is composite of chunks as tuples
                        # instead of int
                        data_src = data.rechunk(data_dst.chunks)
                        array = data_src.to_zarr(
                            data_dst,
                            overwrite=overwrite,
                            compute=False,
                            return_stored=True,
                        )

                        @delayed
                        def calc_checksum(array):
                            # NOTE cannot create external dependency in delayed funcs,
                            # map futures to their target groups instead
                            return xxhash.xxh64(array).hexdigest()

                        checksum = calc_checksum(array)
                        future = client.compute(checksum)
                        futures[future] = label_group
                        # TODO add callback here to update progressbar

        if not futures:
            return  # nothing to do

        n_failed = 0
        for future in as_completed(futures.keys()):
            try:
                group, checksum = futures[future], future.result()
            except Exception:
                logger.error(f'failed to serialize "{group.path}"')
                n_failed += 1
            else:
                logger.debug(f'"{group.path}" xxh64="{checksum}"')
                group.attrs["checksum"] = checksum
        if n_failed > 0:
            logger.error(f"{n_failed} failed serialization task(s)")

    ##

    def _open_session(self):
        try:
            z = zarr.open(self.root_dir, mode="r")  # don't create it
        except ValueError:
            # nothing to open here, unlikely a zarr dataset
            return
        else:
            self._handle = z[self.path]

        # preview the internal structure
        if logger.getEffectiveLevel() <= logging.DEBUG:
            zarr.tree(self._handle)

    def _close_session(self):
        self._handle.close()
        self._handle = None

    def _can_read(self):
        try:
            magic = self.handle.attrs["zarr_dataset"]
            version = self.handle.attrs["format_version"]
        except KeyError:
            return False
        else:
            # verify version
            require_version = type(self).version
            if version != type(self).version:
                logger.debug(
                    f"version mis-match (require: {require_version}, provide: {version})"
                )
                return False

            logger.debug(f"a uToolbox written dataset, {magic} {version}")
            return True

    def _enumerate_files(self):
        #   /time/channel/setup/label/level

        level_str = str(self.level)
        logger.info(f'searching label "{self.label}" (level: {level_str})')

        files = []
        for t, t_root in self.handle.groups():
            for c, c_root in t_root.groups():
                for s, s_root in c_root.groups():
                    if self.label not in s_root:
                        # no data for this spatial setup
                        continue
                    s_root = s_root[self.label]
                    if level_str in s_root:
                        # build path
                        path = f"/{t}/{c}/{s}/{self.label}/{level_str}"
                        files.append(path)
        return files

    def _load_array_info(self):
        shape, dtype = set(), set()
        for array in self.files:
            array = self.handle[array]
            shape.add(array.shape)
            dtype.add(array.dtype)

        inconsist = len(shape) != 1 or len(dtype) != 1
        shape, dtype = next(iter(shape)), next(iter(dtype))
        if inconsist:
            logger.error(
                f"array definition is inconsistent somewhere, using {shape}, {dtype}"
            )

        return shape, dtype

    def _load_channel_info(self):
        return self._load_injective_attributes("channel", required=True)

    def _load_metadata(self):
        dim_info = defaultdict(lambda: defaultdict(list))

        # /time[G]/channel[G]/setup[G]/label[G/A](/level[A])
        groups = ["time", "channel", "setup", "label"]

        def nested_iters(root, groups):
            """
            Loop over each group and extract their raw attributes.
            
            Args:
                root (Group): the root to start with
                indices (list of str): the index list to use as key in `dim_info`
            """
            index = groups[0]
            if len(groups) > 1:
                iterator = root.groups()
            else:
                # last dimension is label, can be
                #   - simple, array, label[A]
                #   - multiscale, group, label[G]/level[A]
                if isinstance(root, zarr.Group):
                    iterator = root.arrays()
                
                # TODO refactor from here below
                iterator = root.arrays()
            for name, child in iterator:
                dim_info[index][name].append(child.attrs)  # lazy load
                if len(indices) > 1:
                    nested_iters(child, indices[1:])

            # TODO factor in label group/array differences

        nested_iters(self.handle, groups)

        print(dim_info)
        raise RuntimeError("DEBUG")

        return dim_info

    def _load_mapped_coordinates(self):
        # NOTE we store index/coord under the same setup attrs, therefore, we construct
        # the mapped coordinate table directly
        coords, index = [], []
        for key, attrs in self.metadata["setup"].items():
            for attr in attrs:
                if "tile_coord" in attr:
                    coords.append(attr["tile_coord"])
                if "tile_index" in attr:
                    index.append(attr["tile_index"])

        coords, index = pd.DataFrame(coords), pd.DataFrame(index)

        # rename index (which is a bit annoy to rename as multi-index)
        index_names_mapping = {ax: f"tile_{ax}" for ax in index.columns}
        index.rename(index_names_mapping, axis="columns", inplace=True)

        # sanity check
        if len(coords) != len(index):
            logger.error("coordinates and index info mismatch")
            index = self._infer_index_from_coords(coords)

        # build multi-index
        df = pd.concat([index, coords], axis="columns")
        df.set_index(index.columns.to_list(), inplace=True)

        # rename coords
        coord_names_mapping = {ax: f"{ax}_coord" for ax in coords.columns}
        df.rename(coord_names_mapping, axis="columns", inplace=True)

        return df

    def _load_timestamps(self) -> List[np.datetime64]:
        return self._load_injective_attributes("time")

    def _load_view_info(self):
        return self._load_injective_attributes("view")

    def _load_injective_attributes(self, dim_name, required=False):
        """
        Load group dimensional attributes that are 1-1 mapping across different group 
        names.

        Args:
            dim_name (str): dimension name to extract
            required (bool, optional): the attribute must exist
        """
        mapping = defaultdict(set)
        for key, attrs in self.metadata[dim_name].items():
            for attr in attrs:
                try:
                    value = attr[dim_name]
                except KeyError:
                    if required:
                        raise KeyError(f'"{key}" does not have attribute "{dim_name}"')
                else:
                    # NOTE split to try-except-else to ensure we do not create
                    # unncessary keys
                    mapping[key].add(value)

        for key, value in mapping.items():
            inconsist = len(value) > 1
            value = next(iter(value))
            if inconsist:
                logger.error(
                    f'"{key}" has inconsistent attribute definitions, using "{value}"'
                )
            mapping[key] = value

        # NOTE duing `_update_inventory_index`, we rely on None to determine columns to
        # drop
        attrs = list(mapping.values())
        return attrs if attrs else None

    def _load_voxel_size(self):
        voxel_size = set()
        for array in self.files:
            group = os.path.dirname(array)
            group = self.handle[group]
            voxel_size.add(tuple(group.attrs["voxel_size"]))

        inconsist = len(voxel_size) > 1
        voxel_size = next(iter(voxel_size))
        if inconsist:
            logger.error(f"voxel size varies, using {voxel_size}")

        # TODO factor in resolution level
        logger.warning("voxel size is native resolution")

        return voxel_size

    def _retrieve_file_list(self, coord_dict):
        print(coord_dict)  # TODO lookup attributes -> group number idF
        raise RuntimeError("DEBUG, _retrieve_file_list")
