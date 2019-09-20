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
        tag, array = self.data["Lattice"]
        with open(path, "r") as fd:
            # search until tag
            for line in fd:
                if line.startswith(tag):
                    break
            # start parsing...
            for i, line in enumerate(fd):
                line = line.strip()
                if not line:
                    continue
                array[i, :] = np.array(
                    [float(v) for v in line.split(" ")], dtype=array.dtype
                )

        # overwrite data
        logger.debug(f"replace internal data as a {array.shape} array")
        self._data = array

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]


if __name__ == "__main__":
    cm = AmiraColormap("pureGreen.col")
    print(cm[4])
