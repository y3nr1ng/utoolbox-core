from ..format import Format
from .. import formats


class Zarr(Format):
    def can_read(self, uri):
        pass

    def can_write(self, uri):
        pass

    class Reader(Format.Reader):
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

    class Writer(Format.Writer):
        pass


# register
format = Zarr("zarr", "Zarr directory-based dataset")
formats.add_format(format)
