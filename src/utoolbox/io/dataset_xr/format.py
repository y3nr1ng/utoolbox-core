from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple


logger = logging.getLogger("utoolbox.io.dataset.format")


class Format(ABC):
    """
    Represents an implementation to read/write a particular dataset format.

    Args:
        name (str): short name of this dataset format
        description (str): one-line description of the format
    """

    def __init__(self, name, description):
        self._name = name
        self._description = description

    def __repr__(self):
        # short description
        return f"<Dataset {self.name} - {self.description}>"

    ##

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def modes(self) -> Tuple[str]:
        return self._modes

    ##

    def get_reader(self, uri: Path, **kwargs):
        return self.Reader(self, uri, **kwargs)

    def get_writer(self, uri: Path, **kwargs):
        return self.Writer(self, uri, **kwargs)

    @abstractmethod
    def can_read(self, uri: Path) -> bool:
        """
        Whether this dataset can read data from the specified uri.

        Args:
            uri (Path): path to the dataset
        """

    @abstractmethod
    def can_write(self, uri: Path) -> bool:
        """
        Whether this dataset can write data to the specified uri.

        Args:
            uri (Path): path to the dataset
        """

    ##

    class BaseReaderWriter(ABC):
        """
        Base class for the Reader/Writer class to implement common context managed
        functions.
        """

        def __init__(self, format: Format, uri: Path, **kwargs):
            self._format = format
            self._uri = Path(uri)

            # is this reader/writer op already terminated?
            self._closed = False

            # open the dataset
            self.open(**kwargs)  # TODO move open() to __enter__

        def __enter__(self):
            self._assert_closed()
            return self

        def __exit__(self, *exc):
            self._close()  # use the wrapped close

        ##

        @property
        def closed(self) -> bool:
            """Whether the reader/writer is closed."""
            return self._closed

        @property
        def format(self) -> Format:
            """The dataset object corresponding to current read/write operation."""
            return self._format

        @property
        def uri(self) -> Path:
            """The uri to dataset corresponding to current read/write operation."""
            return self._uri

        ##

        @abstractmethod
        def open(self, **kwargs):
            """
            It is called when the reader/writer is created. Dataset accessor do its
            initialization here in order to granted reader/writer proper environment to
            work with.
            """

        @abstractmethod
        def close(self):
            """
            Called when the reader/writer is closed.
            """

        ##

        def get_index(self):
            pass

        @abstractmethod
        def set_index(self, **index):
            """
            Set the internal pointer such that the next to :func:`.get_next_data()` 
            returns the data specified this index.
            
            Args:
                **index : TBD
            """

        ##

        def _assert_closed(self):
            if self.closed:
                raise RuntimeError(f"{self.uri} is already closed")

        def _close(self):
            """
            Wrapper function for the actual :func:`.close`. Therefore, close has no
            effect if it is already closed.
            """
            if self.closed:
                return

            self._closed = True
            self.close()

    class Reader(BaseReaderWriter):
        """
        The purpose of a reader object is to read data from a dataset resource, and 
        should be obtained by calling :func:`.get_reader`.
        """

        def __iter__(self):
            self._assert_closed()
            # TODO loop over all the data by setting index sequentially

        @abstractmethod
        def __len__(self):
            """Get the number of data in the dataset."""

        ##

        @abstractmethod
        def get_data(self, **index):
            """
            Read data from the dataset using provided multi-dimensional index.

            Args:
                **index : TBD
            """

        @abstractmethod
        def get_next_data(self):
            """
            Return the next data from the series.

            TODO how to specify dimensional order?
            """
            pass

        @abstractmethod
        def get_metadata(self, **index):
            """
            Read metadata of the data at provided index. If the index is None, this returns the global metadata.

            Args:
                **index : TBD
            """
            pass

    class Writer(BaseReaderWriter):
        """
        The purpose of a writer object is to write data to a dataset resource, and 
        should be obtained by calling :func:`.get_writer`.
        """

        @abstractmethod
        def set_data(self, data, **index):
            pass

        @abstractmethod
        def append_data(self, data):
            pass

        @abstractmethod
        def set_metadata(self, data: Dict[str, Any]):
            pass


class ZarrDataset(Dataset):
    def can_read(self, uri):
        pass

    def can_write(self, uri):
        pass

    class Reader(Dataset.Reader):
        def __len__(self):
            pass

        ##

        def open(self, **kwargs):
            pass

        def close(self):
            pass

        ##

        def get_data(self, **index):
            pass

        def get_next_data(self):
            pass

        def get_metadata(self, **index):
            pass

        def set_index(self, **index):
            pass

    class Writer(Dataset.Writer):
        pass


class FormatManager:
    def __init__(self):
        self._formats = []

    def __repr__(self):
        return f"<FormatManager, {len(self)} registered formats>"

    def __iter__(self):
        return iter(self._formats)

    def __len__(self):
        return len(self._formats)

    ##

    def add_format(self, format, overwrite=False):
        if not isinstance(format, Format):
            pass

    def search_read_format(self, uri):
        """
        Search a format that can read the uri.

        Args:
            uri (Path): path to the dataset
        """
        for f in self._formats:
            if f.can_read(uri):
                return f

    def search_write_format(self, uri):
        """
        Search a format that can write the uri.

        Args:
            uri (Path): path to the dataset
        """
        for f in self._formats:
            if f.can_write(uri):
                return f

    def get_format_names(self) -> Tuple[str]:
        return tuple(f.name for f in self)

    def show(self):
        """Show formatted list of available formats."""
        raise NotImplementedError
