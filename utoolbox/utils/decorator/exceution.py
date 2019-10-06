"""
Decorators related to reshaping program execution flows.
"""
__all__ = ["lazy_property", "run_once"]


class lazy_property(object):
    """Monkey patch the wrapped function after evaluation."""

    def __init__(self, func):
        self._func = func

    def __get__(self, instance, owner):
        """
        :param object intance: instance that the attribute was accessed through
        :param object owner: owner class of the instance
        
        :return: evaluated value from encapsulated instance method
        """
        value = self._func(instance)
        setattr(instance, self._func.__name__, value)
        return value


class run_once(object):
    """
    Both unbound and bound methods will run only once.

    Reference
    ---------
    http://code.activestate.com/recipes/425445-once-decorator/
    """

    __slots__ = ("_func", "_result", "_methods")

    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kargs):
        try:
            return self._result
        except AttributeError:
            self._result = self._func(*args, **kargs)
            return self._result

    def __get__(self, instance, owner):
        method = self._func.__get__(instance, owner)
        try:
            return self._methods[method]
        except (AttributeError, KeyError):
            decorated = run_once(method)
            try:
                self._methods[method] = decorated
            except AttributeError:
                self._methods = {method: decorated}
            return decorated

    def __eq__(self, other):
        return isinstance(other, run_once) and other._func == self._func

    def __hash__(self):
        return hash(self._func)
