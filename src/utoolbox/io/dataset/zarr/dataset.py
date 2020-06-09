import logging
import os
from collections import defaultdict
from itertools import chain
from typing import List, Optional

import dask.array as da
import numpy as np
import pandas as pd
import xxhash
import zarr
from dask import delayed
from dask.distributed import as_completed

from utoolbox.util.dask import get_client

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
        strict (bool, optional): resolution level should exist instead of fallback
    """

    def __init__(
        self, store: str, label: str = RAW_DATA_LABEL, level: int = 0, path: str = "/"
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
        if level < 0:
            raise ValueError("level should >= 0")
        self._level = level

        # TODO how to reload dataset from filesystem
        raise NotImplementedError

    @property
    def read_func(self):
        def func(uri, shape, dtype):
            group = self.handle[uri]

            # multi-scale?
            if ZarrDataset.is_multiscales(group):
                name = os.path.basename(uri)
                for group_attrs in group.attrs["multiscales"]:
                    # find the corresponding attributes
                    if group_attrs["name"] == name:
                        # get path for requested level
                        level = group_attrs["datasets"][self.level]["path"]
                        break
                else:
                    raise RuntimeError(
                        f'corrupted multiscale dataset, unable to find path for "{name}" (level: {self.level})'
                    )
            else:
                level = ""

            # build final path
            # NOTE "zarr.Group + sub-path" does not function properly, use "str + full
            # path" instead
            path = "/".join([uri, level])

            # zarr array contains shape, dtype and decompression info
            return da.from_zarr(self.root_dir, path)

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
                    for (st_index, st_coord), selected in tiles_iterator:
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
                        if st_index is None:
                            for index in ("tile_index", "tile_coord"):
                                try:
                                    del s_root.attrs[index]
                                except KeyError:
                                    pass
                        else:
                            names = tiles_iterator.index
                            names = [name.split("_")[-1] for name in names]
                            # NOTE numpy dtypes are not serializable, use native
                            s_root.attrs["tile_index"] = {
                                k: int(v) for k, v in zip(names, st_index)
                            }
                            s_root.attrs["tile_coord"] = {
                                k: float(v) for k, v in zip(names, st_coord)
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

    @staticmethod
    def is_multiscales(group, strict=False):
        if not isinstance(group, zarr.Group):
            return False

        try:
            attrs = group.attrs["multiscales"]
        except KeyError:
            return False

        # NOTE multiscales is designed to contain multiple multiscale-dataset, scan
        # over them
        name = os.path.basename(group.name)
        for group_attrs in attrs:
            if group_attrs["name"] == name:
                attrs = group_attrs
                break
        else:
            return False

        message = None
        try:
            if attrs["version"] != "0.1":
                message = "multiscale version mismatch"
            elif attrs["name"] != name:
                message = "multiscale attribute does not belong to this array"
            elif any(info["path"] not in group for info in attrs["datasets"]):
                message = "multiscale dataset is damaged, missing dataset"
        except KeyError:
            message = "multiscale dataset is damaged, missing key"
        if message:
            if strict:
                raise ValueError(message)
            else:
                logger.warning(message)

        return True

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
        logger.info(f'searching for "{self.label}" (level: {level_str})')

        files = []
        for t, t_root in self.handle.groups():
            for c, c_root in t_root.groups():
                for s, s_root in c_root.groups():
                    if self.label not in s_root:
                        # no data for this spatial setup
                        continue
                    s_root = s_root[self.label]

                    path = f"/{t}/{c}/{s}/{self.label}"
                    if level_str in s_root:
                        # multi-level dataset
                        path = f"{path}/{level_str}"
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
        channels, self._channel_id_lut = self._load_injective_attributes(
            "channel", required=True
        )
        return channels

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
            if len(groups) > 1:
                iterator = root.groups()
            else:
                iterator = chain(root.groups(), root.arrays())

            for name, child in iterator:
                index = groups[0]
                dim_info[index][name].append(child.attrs)
                try:
                    nested_iters(child, groups[1:])
                except IndexError:
                    pass

        nested_iters(self.handle, groups)

        return dim_info

    def _load_mapped_coordinates(self):
        # NOTE we store index/coord under the same setup attrs, therefore, we construct
        # the mapped coordinate table directly
        coords, index, group_names = [], [], {}
        for group_name, attrs in self.metadata["setup"].items():
            for group_attrs in attrs:
                if ("tile_coord" in group_attrs) and ("tile_index" in group_attrs):
                    coords.append(group_attrs["tile_coord"])

                    index_ = group_attrs["tile_index"]
                    index.append(index_)
                else:
                    index_ = None
                group_names[group_name] = index_

                # ... otherwise, this setup does not belong to a tile

        # save reverse lookup table
        mapping = defaultdict(list)
        for setup_name, index_ in group_names.items():
            try:
                # ensure dict is sorted in ZYX order
                index_ = dict(sorted(index_.items(), reverse=True))
                index_ = tuple(v for v in index_.values())
            except AttributeError:
                # None, probably not a tile object, therefore, no index tuple
                index_ = None
            mapping[index_].append(setup_name)
        self._tile_id_lut = mapping

        # build dataframe
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
        if df.empty:
            # nothing to build `tile_coords`, use None
            return None
        df.set_index(index.columns.to_list(), inplace=True)

        # rename coords
        coord_names_mapping = {ax: f"{ax}_coord" for ax in coords.columns}
        df.rename(coord_names_mapping, axis="columns", inplace=True)

        return df

    def _load_timestamps(self) -> List[np.datetime64]:
        timestamps, self._timestamp_id_lut = self._load_injective_attributes(
            "time", "timestamp"
        )
        return timestamps

    def _load_view_info(self):
        views, self._view_id_lut = self._load_injective_attributes("setup", "view")
        return views

    def _load_injective_attributes(self, dim_name, key=None, required=False):
        """
        Load group dimensional attributes that are 1-1 mapping across different group
        names.

        Args:
            dim_name (str): dimension name to extract
            key (str, optional): key name to search in the attributes
            required (bool, optional): the attribute must exist

        Returns:
            (Tuple[List[values], Mapping[reverse lookup]])
        """
        key = dim_name if key is None else key

        mapping = defaultdict(set)
        for group_name, attrs in self.metadata[dim_name].items():
            for attr in attrs:
                try:
                    value = attr[key]
                except KeyError:
                    if required:
                        raise KeyError(
                            f'"{group_name}" does not have attribute "{key}"'
                        )
                    value = None
                # NOTE split to try-except-else to ensure we do not create unnecessary
                # keys
                mapping[group_name].add(value)

        for key, value in mapping.items():
            inconsist = len(value) > 1
            value = next(iter(value))
            if inconsist:
                logger.error(
                    f'"{key}" has inconsistent attribute definitions, using "{value}"'
                )
            mapping[key] = value

        # NOTE during `_update_inventory_index`, we rely on `None` to determine columns
        # to drop
        attrs = list(set(mapping.values()))
        if attrs and (attrs[0] is not None):
            attrs.sort()
        else:
            attrs = None  # compress all the None-s into None

        # generate reverse mapping (used in retrieve file list)
        rmapping = defaultdict(list)
        for k, v in mapping.items():
            rmapping[v].append(k)
        # simplify 1 item lists
        rmapping = {k: v if len(v) > 1 else v[0] for k, v in rmapping.items()}

        return attrs, rmapping

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
        # t
        time = self._timestamp_id_lut[coord_dict.get("time", None)]
        # c
        channel = self._channel_id_lut[coord_dict.get("channel")]
        # NOTE setups are mostly _not_ 1:1, therefore, we need to use `set` to filter
        # them out
        # s, view
        view = self._view_id_lut[coord_dict.get("view", None)]
        view = set(
            [view] if isinstance(view, str) else view
        )  # ensure it is an iterable
        # s, tile
        tile_index = (
            tuple(coord_dict[index] for index in self.tile_index_str)
            if self.tile_index_str
            else None
        )
        tile = set(self._tile_id_lut[tile_index])
        # s
        setup = view & tile
        assert len(setup) == 1, "unable to determine an unique setup, programmer error"
        setup = next(iter(setup))

        path = "/".join(["", time, channel, setup, self.label])
        return path


class MutableZarrDataset(ZarrDataset):
    @classmethod
    def from_immutable(dataset: ZarrDataset):
        """From immutable Zarr dataset."""
        pass

    ##

    def attach_data(self, uuid, label, data, attrs=None):
        """
        Attach additional data to setup.

        Args:
            uuid (str): uuid of the source data group to attach
            label (str): label for the new data
            data : the data to store in the group
            attrs (dict, None): attributes for the data
        """
        pass

    def delete_data(self, uuid, label=None):
        pass

    ##

    def to_multiscale(self, uuid, levels, label=None):
        """
        Convert a typical dense data array to Zarr multiscale format.

        Args:
            uuid (str):
            levels (list of tuple of int):
            label (str):
        """
        pass

    def reduce_multiscale(self, uuid, label=None, keep_level=0):
        """
        Reduce multiscale to highest resolution.

        If label is not specified, it will use the active label group selected during 
        dataset creation. Default to keep level with the highest resolution.

        Args:
            uuid (str): uuid of the source data group
            label (str, optional): label to operate on
            keep_level (int, optional): the level to keep
        """
        pass
