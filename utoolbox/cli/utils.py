from functools import wraps


def processor(func):
    """
    Helper decorator to rewrite a function so that it returns another function from it.
    
    Adapt from clicks `imagepipe` example.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        return func(dataset, *args, **kwargs)

    return wrapped_func


def generator(func):
    """Helper function that continuously generate data source for others to ingest."""
    pass
