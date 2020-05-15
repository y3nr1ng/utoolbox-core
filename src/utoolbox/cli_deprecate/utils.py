from functools import update_wrapper


def processor(func):
    """
    Helper decorator to rewrite a function so that it returns another function from it.
    
    Adapt from clicks `imagepipe` example.
    """

    def new_func(*args, **kwargs):
        def processor(stream):
            return func(stream, *args, **kwargs)

        return processor

    return update_wrapper(new_func, func)


def generator(func):
    """Helper function that continuously generate data source for others to ingest."""

    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in func(*args, **kwargs):
            yield item

    return update_wrapper(new_func, func)
