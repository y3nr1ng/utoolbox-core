from typing import Optional

from .format import Format


class Dataset:
    def __init__(
        self, reader: Optional[Format.Reader], writer: Optional[Format.Writer]
    ):
        self._reader = reader
        self._writer = writer

    ##

    @property
    def reader(self) -> Format.Reader:
        if self._reader is None:
            raise RuntimeError("dataset is not readable")
        return self._reader

    @property
    def writer(self) -> Format.Writer:
        if self._writer is None:
            raise RuntimeError("dataset is not writable")
        return self._writer
