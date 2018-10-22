from abc import ABCMeta, abstractmethod

class Algorithm(object):
    """Common interface of interest."""

    def __init__(self, *args, use_gpu=True, **kwargs):
        self._strategy = self.GPU(self) if use_gpu else self.CPU(self)

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *exc_args):
        self.deinit()

    def __call__(self, *args, **kwargs):
        return self._strategy(*args, **kwargs)

    def init(self):
        self._strategy.init()

    def deinit(self):
        self._strategy.deinit()

    class Context(metaclass=ABCMeta):
        def __init__(self, parent):
            self.parent = parent

        @abstractmethod
        def __call__(self, *args, **kwargs):
            raise NotImplementedError

        def init():
            pass

        def deinit():
            pass

    class GPU(Algorithm.Context):
        pass

    class CPU(Algorithm.Context):
        pass
