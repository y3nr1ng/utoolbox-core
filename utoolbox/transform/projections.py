import logging
from math import ceil

import cupy as cp
from cupy.cuda.memory import OutOfMemoryError


__all__ = ["Orthogonal"]

logger = logging.getLogger(__name__)


class Orthogonal(object):
    """
    Orthogonal maximum intensity projections.
    
    Args:
        data (np.array or cp.array): volume to project
        block_sz (tuple of int): kernel block size
    """

    def __init__(self, data, mode="max", block_sz=(16, 16)):
        self._mode = mode
        self._block_sz = block_sz

        try:
            self._data = cp.asarray(data)
            self._project = self._project_whole
        except OutOfMemoryError:
            logging.info("unable to fit in memory, slicing...")
            # process data in slabs
            self._as_slab = True
            # slice data along z until it can fit in the memory
            nz, factor = data.shape[0], 2
            while True:
                _nz = ceil(nz / factor)
                try:
                    self._data = cp.asarray(data[:_nz, ...])
                except OutOfMemoryError:
                    if _nz == 1:
                        raise OutOfMemoryError("data is horrendously large")
                    factor *= 2
                else:
                    logging.info(f".. z slicing factor {factor}")
                    self._step = _nz
                    break

            self._data = data
            self._project = self._project_slab

    ##

    def _project_whole(self, axis):
        func = getattr(self._data, self._mode)
        return cp.asnumpy(func(axis=axis))

    def _project_slab(self, axis):
        proj = []
        for data in self._load_slab():
            func = getattr(data, self._mode)
            proj.append(func(axis=axis))
            # release the reference in this loop
            func = None
            data = None
        if axis == 0:
            # z is an aggregated axis, sum together
            proj = sum(proj)
        else:
            # z is a visible axis, stack together
            proj = cp.vstack(proj)
        return cp.asnumpy(proj)

    def _load_slab(self):
        nz = self._data.shape[0]

        # segment ranges
        start = list(range(0, nz, self._step))
        end = start[1:] + [nz]

        for z0, z1 in zip(start, end):
            logger.debug(f".. z [{z0}, {z1})")

            tmp = cp.asarray(self._data[z0:z1, ...])
            yield tmp
            tmp = None

    ##

    @property
    def xy(self):
        logger.debug("XY")
        return self._project(axis=0)

    @property
    def xz(self):
        logger.debug("XZ")
        return self._project(axis=1)

    @property
    def yz(self):
        logger.debug("YZ")
        return self._project(axis=2)
