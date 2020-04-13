import logging

import numpy as np
import pandas as pd

from utoolbox.data.io.amira.base import Amira

__all__ = ["AmiraPointCloud"]

logger = logging.getLogger(__name__)


class AmiraPointCloud(Amira):
    def __init__(self, path):
        super().__init__(path)

        sid_ranges = self._extract_sid_ranges()

        damaged = False

        arrays = dict()
        for sid, (start, end) in sid_ranges:
            # find data type
            name, dtype = None, None
            for _name, (_sid, shape, _dtype) in self.data.items():
                if sid == _sid:
                    name, dtype = _name, _dtype
                    break
            logger.info(f'loading "{name}" from "{path}" (offset: {start})')
            # load from file
            count = (end - start) // np.dtype(dtype).itemsize
            try:
                array = (
                    np.fromfile(path, dtype=dtype, count=count, offset=start)
                    .reshape(shape)
                    .squeeze()
                )
            except ValueError:
                # only fix last dimension
                entries = count // shape[-1]
                shape = (entries, shape[-1])
                # reduce overall elements
                _count = entries * shape[-1]
                logger.error(f"array size coerced from {count} to {_count} {shape}")
                # reshape again
                array = (
                    np.fromfile(path, dtype=dtype, count=_count, offset=start)
                    .reshape(shape)
                    .squeeze()
                )
                damaged = True
            arrays[name] = array

        if damaged:
            arrays["Ids"] = arrays["Ids"][:-1]

        df_source = {"Point ID": arrays["Ids"]}
        for i, ax in enumerate(["X", "Y", "Z"]):
            df_source[f"{ax} Coord"] = arrays["Coordinates"][:, i]
        df = pd.DataFrame(df_source)

        self._data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __getattr__(self, name):
        return getattr(self.data, name)

    ##

    def _extract_sid_ranges(self):
        sid_list = [sid for sid, _, _ in self.data.values()]
        sid_pos = []
        with open(self.path, "r", errors="ignore") as fd:
            # skip to data section
            line = fd.readline()
            while not line.startswith("# Data section follows"):
                line = fd.readline()
            # start looking
            # NOTE manually iterate to avoid next() from disabling tell()
            line = fd.readline()
            while line:
                for sid in sid_list:
                    if line.startswith(sid):
                        offset = fd.tell() - len(sid) - 1  # include 1 LF character
                        logger.debug(f'found "{sid}" at offset {offset}')
                        sid_pos.append((sid, offset))
                        sid_list.remove(sid)
                        break
                line = fd.readline()
            # .. add EOF marker
            fd.seek(0, 2)
            sid_pos.append(("end", fd.tell()))  # include 1 LF character
        # convert to range
        # .. sort by position in the file
        sid_pos.sort(key=lambda x: x[1])
        # .. iterate over as tuple
        sid_ranges = [
            (sid, (start + len(sid) + 1, end - 1))
            for (sid, start), (_, end) in zip(sid_pos[:-1], sid_pos[1:])
        ]

        return sid_ranges

    def _validate_file(self):
        ctype = self.metadata["parameters"]["ContentType"]
        if ctype != "HxCluster":
            raise RuntimeError("not an Amira PointCloud")
