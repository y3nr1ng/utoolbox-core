class FileIO(object):
    def __init__(self, name, mode):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError

    def __len__(self):
        """Return number of subfiles."""
        raise NotImplementedError

    def __iter__(self):
        raise TypeError("Not supported.")

    def __next__(self):
        raise TypeError("Not supported.")

    def __getitem__(self, index):
        raise TypeError("Not supported.")

    def __setitem__(self, index, data):
        raise TypeError("Not supported.")

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError
