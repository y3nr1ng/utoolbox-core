__all__ = ["guess_data_root"]


def guess_data_root(data_dir="data"):
    """Guess the data directory that follows this package."""
    import inspect
    import os

    import utoolbox

    module_path = inspect.getmodule(utoolbox).__path__._path[0]
    return os.path.join(module_path, data_dir)
