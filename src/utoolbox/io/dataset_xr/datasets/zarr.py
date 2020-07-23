from .. import formats
from ..request import Request
from ..format import Format


class Zarr(Format):
    def can_read(self, request: Request):
        return True

    def can_write(self, request: Request):
        return True

    class Reader(Format.Reader):
        def __len__(self):
            pass

        ##

        def open(self, **kwargs):
            print("reader open")

        def _close(self):
            pass

        ##

        def set_index(self, **index):
            pass

        ##

        def get_data(self, **index):
            pass

        def get_metadata(self, **index):
            pass

    class Writer(Format.Writer):
        def open(self, **kwargs):
            print("writer open")

        def _close(self):
            pass

        ##

        def set_index(self, **index):
            pass

        ##

        def set_data(self):
            pass

        def set_metadata(self):
            pass


# register
format = Zarr("zarr", "Zarr directory-based dataset")
formats.add_format(format)
