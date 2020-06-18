from prefect import Task
import logging

__all__ = ["ZarrWriteArray"]

logger = logging.getLogger("utoolbox.pipeline.tasks")


class ZarrWriteArray(Task):
    def __init__(self, rechunk="auto", **kwargs):
        super().__init__(name="zarr-write-array", **kwargs)

        self._rechunk = rechunk

    def run(self, src, dst, **kwargs):
        """
        Args:
            TBA
            src (dask.array)
            dst (zarr.array)
        """
        if self._rechunk == "auto":
            src_ = src.rechunk(dst.chunks)
        elif self._rechunk is not None:
            src_ = src.rechunk(src)
        else:
            src_ = src
        return src_.to_zarr(dst, **kwargs)
