class FileIO(object):
    def __init__(self, name, mode='r'):
        raise NotImplemented

    def __enter__(self):
        raise NotImplemented

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplemented
