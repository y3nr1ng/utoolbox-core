import glob
from io import StringIO
import logging
import os

import pandas as pd

from ..base import DenseDataset, MultiChannelDataset, MultiViewDataset, TiledDataset

from .error import MissingMetadataError

__all__ = ["SmartSpimDataset"]

logger = logging.getLogger(__name__)


class SmartSpimDataset(
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

    def _can_read(self):
        return bool(self.metadata)

    def _enumerate_files(self):
        # SmartSPIM uses 3 level structure
        search_path = os.path.join(self.root_dir, "*", "*", "*", "*.raw")
        return glob.glob(search_path)

    def _load_array_info(self):
        pass

    def _load_channel_info(self):
        return self.metadata["channels"]["Power"].values

    def _retrieve_file_list(self, coord_dict):
        pass

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

    def _load_tiling_coordinates(self):
        pass

    def _load_tiling_info(self):
        pass

    def _load_view_info(self):
        return self.metadata["coords"]["Side"].unique().values
