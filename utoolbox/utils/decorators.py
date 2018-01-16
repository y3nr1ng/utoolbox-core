def lazy_property(func):
    """Decorator that makes a property lazy-evaluated."""
    attr_name = '_lazy_' + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

def run_once(func):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            print('{} executed'.format(func))
            return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper
