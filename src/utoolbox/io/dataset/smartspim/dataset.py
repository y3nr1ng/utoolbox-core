import glob
from io import StringIO
import logging
import os

from dask import delayed
import dask.array as da
import numpy as np
import pandas as pd

from ..base import DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset

from .error import MissingMetadataError

__all__ = ["SmartSpimDataset"]

logger = logging.getLogger("utoolbox.io.dataset")


class SmartSpimDataset(
    DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset
):
    def __init__(self, root_dir):
        super().__init__()

        self._root_dir = root_dir

    ##

    @property
    def read_func(self):
        @delayed(pure=True, traverse=False)
        def read_raw(uri, shape, dtype):
            data = np.fromfile(uri, dtype=dtype, offset=8)  # skip 2x uint32 (8 bytes)
            return data.reshape(shape)

        def func(uri, shape, dtype):
            # order by z
            uri.sort()

            # layered volume
            nz, shape = shape[0], shape[1:]
            array = da.stack(
                [
                    da.from_delayed(read_raw(file_path, shape, dtype), shape, dtype)
                    for file_path in uri
                ]
            )
            if array.shape[0] != nz:
                logger.warning(f"retrieved layer mis-matched")
            return array

        return func

    @property
    def root_dir(self):
        return self._root_dir

    ##

    def _can_read(self):
        return bool(self.metadata)

    def _enumerate_files(self):
        # SmartSPIM uses 3 level structure
        search_path = os.path.join(self.root_dir, "*", "*", "*", "*.raw")
        return glob.glob(search_path)

    def _load_array_info(self):
        test_path = self.files[0]

        # array shape
        shape = np.fromfile(test_path, dtype=np.uint32, count=2)
        shape = tuple(shape[::-1])

        # number of files in a folder as layers
        parent, _ = os.path.split(test_path)
        file_list = glob.glob(os.path.join(parent, "*.raw"))
        nz = len(file_list)
        shape = (nz,) + shape

        return shape, np.uint16

    def _load_channel_info(self):
        # index
        i_ch = self.metadata["coords"]["Laser"].unique()
        # wavelength
        n_ch = self.metadata["channels"]["Power"].values

        # scan for inconsistencies
        dir_names = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        if len(dir_names) != len(i_ch):
            raise RuntimeError("metadata does not completely describe experiment data")
        for dir_name in dir_names:
            w = int(dir_name.split("_")[1])
            err = [abs(w - w0) for w0 in n_ch]
            if any([e == 0 for e in err]):
                # has perfect match
                continue
            else:
                # search for closest match
                i = err.index(min(err))
                logger.warning(f'remap channel "{n_ch[i]}" to "{w}"')
                n_ch[i] = w

        # string names
        return tuple(str(n_ch[i]) for i in i_ch)

    def _load_metadata(self):
        metadata_path = os.path.join(self.root_dir, "metadata.txt")
        if not os.path.exists(metadata_path):
            raise MissingMetadataError()

        # cleanup the file
        with open(metadata_path, "r", encoding="unicode_escape") as fd:
            metadata_raw = StringIO(fd.read())

        # metadata is a TSV
        #   section 1)
        #       obj, h_res, um/pix, z step
        #   section 2)
        #       power, power (%), balance
        #   section 3)
        #       x, y, z, laser, side, exposure

        # sections and their labels
        #   `None`, will expand the entire section as key/value pairs
        sections = [None, "channels", "coords"]
        keywords = ["Obj\t", "Power\t", "X\t"]

        # find section offsets
        offsets = []
        for i, line in enumerate(metadata_raw):
            _kw = None
            for keyword in keywords:
                if line.startswith(keyword):
                    # found a section
                    offsets.append(i)
                    _kw = keyword
                    break
            else:
                # not a section header
                continue
            # drop from the pending list
            keywords.remove(_kw)
            # early stop
            if not keywords:
                break

        # iterate over sections and parse them as TSV
        metadata = dict()
        offsets += [None]
        for section, start, end in zip(sections, offsets[:-1], offsets[1:]):
            metadata_raw.seek(0)
            if end:
                nrows = end - start - 1
            else:
                # None = read in rest of the data
                nrows = end
            df = pd.read_csv(metadata_raw, sep="\t", skiprows=start, nrows=nrows)
            df.dropna(how="all", axis="columns", inplace=True)

            if section:
                metadata[section] = df
            else:
                # typical summaries, remap as dict
                metadata.update({k: v for k, v in zip(df.columns, df.iloc[0])})

        return metadata

    def _load_coordinates(self):
        # SmartSpim bookkeeps all 3 axis, but only stored in 2-level hierarchy
        coords = self.metadata["coords"][["X", "Y"]].copy()
        coords.rename({"X": "tile_x", "Y": "tile_y"}, axis="columns", inplace=True)

        # convert unit from 1/10 micron to micron
        coords.loc[:, ["tile_x", "tile_y"]] /= 10

        return coords

    def _load_view_info(self):
        return self.metadata["coords"]["Side"].unique()

    def _load_voxel_size(self):
        dxy, dz = self.metadata["µm/pix"], self.metadata["Z step (µm)"]
        return (dz, dxy, dxy)

    def _retrieve_file_list(self, coord_dict):
        # restore to 1/10 micron
        x, y = int(coord_dict["tile_x"] * 10), int(coord_dict["tile_y"] * 10)

        return glob.glob(
            os.path.join(
                self.root_dir,
                f"Ex_{coord_dict['channel']}_*",
                str(x),
                f"{x}_{y}",
                "*.raw",
            )
        )
