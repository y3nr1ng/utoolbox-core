import logging

import numpy as np

from utoolbox.data.io.amira.base import Amira

__all__ = ["AmiraColormap"]

logger = logging.getLogger(__name__)


class AmiraColormap(Amira):
    """Load Amira ASCII colormap format."""

    def __init__(self, path):
        super().__init__(path)

        # load data stream
        if len(self.data) > 1:
            logger.warning("colormap format should contain 1 data section only")

        _, (sid, shape, dtype) = next(iter(self.data.items()))
        array = np.empty(shape, dtype)

        # search location
        with open(self.path, "r") as fd:
            for line in fd:
                if line.startswith(sid):
                    break
            else:
                raise RuntimeError("unable to find data section marker")
            for i, line in enumerate(fd):
                line = line.strip()
                if not line:
                    continue
                array[i, :] = np.array(
                    [float(v) for v in line.split(" ")], dtype=data.dtype
                )
        self._data = array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]

    ##

    def _validate_file(self):
        ctype = self.metadata["parameters"]["ContentType"]
        if ctype != "Colormap":
            raise RuntimeError("not an Amira colormap")

