import logging
from typing import Optional

import zarr

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
        path (str, optional): group path
    """

    def __init__(self, store: str, path: str = "/"):
        super().__init__(store, path)

    ##

    @property
    def read_func(self):
        pass

    ##

    @classmethod
    def dump(
        cls, store: str, dataset, path: Optional[str] = None, overwrite=False, **kwargs
    ):
        """
        Dump dataset.

        Args:
            store (str): path to the data store
            dataset : serialize the provided dataset
            path (str, optional): internal path
            overwrite (bool, optional): overwrite the dataset if exists
            **kwargs : additional argument for `zarr.open` function
        """
        kwargs["mode"] = "a"
        root = zarr.open(store, **kwargs)

        # test attributes
        root.attrs["zarr_dataset"] = "ZarrDataset"
        root.attrs["format_version"] = cls.version

        if path:
            # nested group
            mode = "w" if overwrite else "w-"
            root = root.open_group(path, mode=mode)

        # start populating the container structure
        #   /time/channel/setup/level
        # welp, i have no idea how to do this cleanly without nested structure
        # 1) time
        for i_t, (t, t_selected) in enumerate(TimeSeriesDatasetIterator(dataset)):
            t_root = root.require_group(f"t{i_t}")
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
                    tiles_iterator = TiledDatasetIterator(v_selected, axis="zyx")
                    for st, selected in tiles_iterator:
                        if st is not None:
                            names = tiles_iterator.index
                            names = [name.split("_")[-1] for name in names]
                            attrs["tile_coord"] = {k: v for k, v in zip(names, st)}

                        s_root = c_root.require_group(f"s{i_s}")
                        i_s += 1
                        s_root.attrs.update(attrs)

                        # 4) level
                        l0_group = s_root.require_group("0")
                        print(l0_group)  # DEBUG
                        data = dataset[selected]
                        # NOTE using default chunk shape after rechunk will cause
                        # problem, since chunk size is composite of chunks as tuples
                        # instead of int
                        data_dst = l0_group.empty_like(
                            "data",
                            data,
                            chunks=True,
                            compression="lz4",
                            compression_opts=dict(acceleration=1),
                        )
                        data_src = data.rechunk(data_dst.chunks)
                        data_dst[...] = data_src[...]

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
            magic = self.handle["zarr_dataset"]
            version = self.handle["format_version"]
        except KeyError:
            return False
        else:
            logger.debug(f"a uToolbox written dataset, {magic} {version}")
            return True

    def _enumerate_files(self):
        pass

    def _load_array_info(self):
        pass

    def _load_channel_info(self):
        pass

    def _load_metadata(self):
        pass

    def _load_tiling_coordinates(self):
        pass

    def _load_view_info(self):
        pass

    def _load_voxel_size(self):
        pass

    def _retrieve_file_list(self, coord_dict):
        pass
