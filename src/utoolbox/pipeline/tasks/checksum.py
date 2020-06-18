import logging

from prefect import Task

__all__ = ["CalcXxHash"]

logger = logging.getLogger("utoolbox.pipeline.tasks")


class CalcXxHash(Task):
    def __init__(self, method="xxh64", return_format="default", **kwargs):
        super().__init__(name="calc-xxhash", **kwargs)

        try:
            import xxhash  # TODO add dependency # delayed import
        except ImportError:
            raise ImportError('requires "xxhash" to use CalcXxHash')

        try:
            self._hash_func = {"xxh64": xxhash.xxh64, "xxh32": xxhash.xxh32}[method]
        except KeyError:
            raise ValueError("unknown xxhash format")

        if return_format not in ("default", "hex", "int"):
            raise ValueError("unknown return format")
        self._return_format = return_format

    def run(self, data):
        try:
            data = data.compute()
        except AttributeError:
            # not a dask array
            pass
        result = self._hash_func(data)

        # TODO elegantly request different digests
        if self._return_format == "default":
            return result.digest()
        elif self._return_format == "hex":
            return result.hexdigest()
        elif self._return_format == "int":
            return result.intdigest()
