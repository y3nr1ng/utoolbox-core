import logging
from pprint import pprint

import numpy as np

from utoolbox.data.io.amira.base import Amira

__all__ = ["AmiraSpatialGraph"]

logger = logging.getLogger(__name__)


class AmiraSpatialGraph(Amira):
    def __init__(self, path):
        super().__init__(path)

        sid_ranges = self._extract_sid_ranges()

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
            arrays[name] = (
                np.fromfile(path, dtype=dtype, count=count, offset=start)
                .reshape(shape)
                .squeeze()
            )

        pprint(arrays)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

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


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    sg = AmiraSpatialGraph("c6_spatialgraph_0042.am")
