import logging
from collections import defaultdict
from typing import List, Optional

import dask.array as da
import numpy as np
import zarr
from dask.distributed import as_completed
from tqdm import tqdm

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
    SessionDataset, DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset
):
    version = 1

    """
    Using Zarr directory as backend. 

    Default internal path is '/', post-processing steps use groups with '_' 
    prefix as their root.

    Args:
        store (str): path to the data store
        label (str, optional): label of the data array
        path (str, optional): internal group path
        level (int, optional): resolution level
    """

    def __init__(
        self, store: str, label: str = RAW_DATA_LABEL, path: str = "/", level: int = 0
    ):
        self._label, self._level = label, 0

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
        Dump dataset.

        Args:
            store (str): path to the data store
            dataset : serialize the provided dataset
            label (str, optional): label of the data array
            path (str, optional): internal path
            overwrite (bool, optional): overwrite the dataset if exists
            client (Client, optional): remote cluster client
            **kwargs : additional argument for `zarr.open` function
        """
        kwargs["mode"] = "a"
        root = zarr.open(store, **kwargs)

        # test attributes
        root.attrs["zarr_dataset"] = "ZarrDataset"
        root.attrs["format_version"] = cls.version

        # TODO save original metadata, msgpack?

        if path:
            # nested group
            mode = "w" if overwrite else "w-"
            root = root.open_group(path, mode=mode)

        tasks = []  # conversion tasks, batch submit after populated all of them

        # start populating the container structure
        #   /time/channel/setup/level
        # welp, i have no idea how to do this cleanly without nested structure
        # 1) time
        for i_t, (t, t_selected) in enumerate(TimeSeriesDatasetIterator(dataset)):
            t_root = root.require_group(f"t{i_t}")

            try:
                # convert from np.datetime64 to int (JSON serializable)
                t = t.to_timedelta64()  # ns
                t = int(t) // 1000000  # ms
            except AttributeError:
                # not a time series
                pass
            t_root.attrs["timestamp"] = t

            # 2) channel
            for i_c, (c, c_selected) in enumerate(
                MultiChannelDatasetIterator(t_selected)
            ):
                c_root = t_root.require_group(f"c{i_c}")
                c_root.attrs["channel"] = c

                # 3) setup
                i_s = 0
                for sv, v_selected in MultiViewDatasetIterator(c_selected):
                    attrs = {}
                    if sv is not None:
                        attrs["view"] = sv
                    tiles_iterator = TiledDatasetIterator(
                        v_selected, axes="zyx", return_format="both"
                    )
                    for st, selected in tiles_iterator:
                        if st is not None:
                            index, coord = st
                            names = tiles_iterator.index
                            names = [name.split("_")[-1] for name in names]
                            # NOTE numpy dtypes are not serializable, use native
                            attrs["tile_index"] = {
                                k: int(v) for k, v in zip(names, index)
                            }
                            attrs["tile_coord"] = {
                                k: float(v) for k, v in zip(names, coord)
                            }

                        s_root = c_root.require_group(f"s{i_s}")
                        i_s += 1
                        s_root.attrs.update(attrs)

                        # 4) level
                        l0_group = s_root.require_group("0")
                        print(l0_group)  # FIXME remove debug
                        data = dataset[selected]
                        # NOTE compression benchmark reference http://alimanfoo.github.
                        # io/2016/09/21/genotype-compression-benchmark.html
                        data_dst = l0_group.empty_like(
                            label,
                            data,
                            chunks=True,
                            compression="blosc",
                            compression_opts=dict(
                                cname="lz4", clevel=5, shuffle=zarr.blosc.SHUFFLE
                            ),
                        )
                        # NOTE using default chunk shape after rechunk will cause
                        # problem, since chunk size is composite of chunks as tuples
                        # instead of int
                        data_src = data.rechunk(data_dst.chunks)
                        task = data_src.to_zarr(
                            data_dst, overwrite=overwrite, compute=False
                        )
                        tasks.append(task)

        def wait_future(futures):
            n_failed = 0
            # TODO should tqdm become built in? dummy class when import error?
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as error:
                    logger.exception(error)
                    n_failed += 1
            if n_failed > 0:
                logger.error(f"{n_failed} task(s) failed")

        # submit the task to cluster
        if client:
            futures = client.compute(tasks)
            wait_future(futures)
        else:
            from dask.distributed import Client

            logger.info("launch a temporary local cluster")
            with Client(
                memory_target_fraction=False,
                memory_spill_fraction=False,
                memory_pause_fraction=0.6,
            ) as client:
                futures = client.compute(tasks)
                wait_future(futures)

    ##

    def _open_session(self):
        z = zarr.open(self.root_dir, mode="r")  # don't create it
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
            if version != type(self).version:
                return False

            logger.debug(f"a uToolbox written dataset, {magic} {version}")
            return True

    def _enumerate_files(self):
        #   /time/channel/setup/level

        l_str = str(self.level)
        files = []
        for t, t_root in self.handle.groups():
            for c, c_root in t_root.groups():
                for s, s_root in c_root.groups():
                    if l_str not in s_root:
                        # no data for this resolution level
                        continue
                    s_root = s_root[l_str]
                    if self.label in s_root:
                        # build path
                        path = f"/{t}/{c}/{s}/{l_str}/{self.label}"
                        files.append(path)
        logger.info(
            f'found {len(files)} file(s) for label "{self.label}", level {self.level}'
        )

        return files

    def _load_array_info(self):
        pass

    def _load_channel_info(self):
        pass

    def _load_metadata(self):
        dim_info = defaultdict(lambda: defaultdict(list))

        # /time[G]/channel[G]/setup[G]/level[G]/label[A]
        indices = ["time", "channel", "setup", "level", "label"]

        def nested_iters(root, indices):
            """
            Loop over each group and extract their raw attributes.
            
            Args:
                root (Group): the root to start with
                indices (list of str): the index list to use as key in `dim_info`
            """
            index = indices.pop(0)
            if len(indices) == 1:
                # last dimension is the array itself
                iterator = root.arrays()
            else:
                iterator = root.groups()
            for name, child in iterator:
                dim_info[index][name].append(child.attrs)  # lazy load
                if indices:
                    nested_iters(child, indices[1:])

        # TODO lowest level is still incorrect

        nested_iters(self.handle, indices)

        return dim_info

    def _load_coordinates(self):
        pass

    def _load_timestamps(self) -> List[np.datetime64]:
        pass

    def _load_view_info(self):
        pass

    def _load_voxel_size(self):
        pass

    def _retrieve_file_list(self, coord_dict):
        pass
