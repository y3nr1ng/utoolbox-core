from timeit import default_timer as timer

def timeit(func):
    """Benchmark the execution time of the wrapped function."""
    def timed(*args, **kwargs):
        t_start = timer()
        result = func(*args, **kwargs)
        t_end = timer()
        print("{} {:2.2f} ms".format(func.__name__, (t_end-t_start) * 1e3))
        return result
    return timed
