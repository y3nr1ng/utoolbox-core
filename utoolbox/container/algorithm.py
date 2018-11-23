from abc import ABCMeta
import enum
import logging
import pprint

try:
    import utoolbox.parallel.gpu
    HAS_GPU_SUPPORT = True
except ImportError:
    HAS_GPU_SUPPORT = False

logger = logging.getLogger(__name__)

__all__ = [
    'ImplTypes',
    'AbstractAlgorithm',
    'interface'
]
class ImplTypes(enum.IntFlag):
    """Types of implementations."""
    CPU_ONLY = 1
    GPU = 2
    DISTRIBUTED = 4

INTERFACE_FLAG = '_interface'

def interface(func):
    """
    This decorate set a label that mark this function to be an requirement in 
    all subclasses.
    """
    setattr(func, INTERFACE_FLAG, True)
    return func

class AbstractAlgorithm(type):
    def __new__(cls, name, bases, dct):
        is_basealgo = '_strategy' not in dct

        if is_basealgo:
            # list of concrete implementations
            dct['_impl'] = dict()

            # required interface functions
            int_funcs = []  
            for func_name, func in dct.items():
                if hasattr(func, INTERFACE_FLAG):
                    int_funcs.append(func_name)
                    #TODO redirect to strategy pattern
            dct['_int_funcs'] = int_funcs

        new_cls = type.__new__(cls, name, bases, dct)

        print("  cls ={}".format(cls))
        print(" name ={}".format(name))
        print("bases ={}".format(bases))
        print("  dct =", end='')
        pprint.pprint(dct)
        print()

        if is_basealgo:
            def __init_subclass__(cls, **kwargs):
                """Register an implementation to its target algorithm."""
                try:
                    super(cls, cls)._impl[cls._strategy] = cls
                except AttributeError:
                    raise AttributeError(
                        "unable to determine implementation type"
                    )
                super().__init_subclass__(**kwargs)
            setattr(
                new_cls, 
                '__init_subclass__', classmethod(__init_subclass__)
            )
        else:
            # confirm interface functions exist
            for func_name in super(new_cls, new_cls)._int_funcs:
                if func_name not in new_cls.__dict__:
                    raise RuntimeError(
                        "incomplete interface, missing \"{}\"".format(func_name)
                    )

        return new_cls

"""
class _AbstractAlgorithm(metaclass=ABCMeta):
    _impls = dict()

    def __init__(self, impl_type):
        self._active_impl_type, self._active_impl = None, None
        self.active_impl_type = impl_type

    def __init_subclass__(cls):
        try:
            FooAlgo_Abstract._impls[cls._strategy] = cls
        except AttributeError:
            raise AttributeError("unable to determine implementation type")

    @property
    def active_impl_type(self):
        return self._active_impl_type

    @active_impl_type.setter
    def active_impl_type(self, impl_type):
        if not isinstance(impl_type, ImplTypes):
            raise TypeError("invalid implementation")

        if impl_type == ImplTypes.GPU:
            if not HAS_GPU_SUPPORT:
                logger.warning("no GPU support, fallback to CPU-only")
                impl_type = ImplTypes.CPU_ONLY

        # force update concrete algorithm
        if self._active_impl_type != impl_type:
            self._active_impl = None
    
        self._active_impl_type = impl_type

    @property
    def active_impl(self):
        if self._active_impl is None:
            #TODO allocate the algorithm
            pass
        return self._active_impl

class FooAlgo_CPU(FooAlgo_Abstract):
    _strategy = ImplTypes.CPU_ONLY

    def __call__(self):
        print("FooAlgo_CPU called")






class ConcreteImpl(type):
    def __new__(cls, name, bases, attrs):
        pass

class Algorithm(metaclass=ABCMeta):
    def __init__(self, impl_type):
        self._active_impl, self.impl_type = None, impl_type

    def __init_subclass__(cls):
        if hasattr(cls, '_strategy'):
            # a concrete strategy
            baseclass = Algorithm._find_baseclass(cls)
            baseclass._register(cls._strategy, cls)
        else:
            # an algorithm definition
            cls._impls = dict()
        super().__init_subclass__()

    @property
    def active_impl(self):
        if self._active_impl is None:
            self._active_impl = self._impls[self.impl_type](
                *self._args, **self._kwargs
            )

    @property
    def impl_type(self):
        return self._impl_type

    @impl_type.setter
    def impl_type(self, impl_type):
        if not isinstance(impl_type, ImplTypes):
            raise TypeError("invalid implementation type")

        if impl_type == ImplTypes.GPU:
            if not HAS_GPU_SUPPORT:
                logger.warning("no GPU support, fallback to CPU-only")
                impl_type = ImplTypes.CPU_ONLY

        self._impl_type = impl_type

    def __call__(self, *args, **kwargs):
        print("cls.impl_type={}".format(str(self.impl_type)))
        self.active_impl(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def _find_baseclass(cls):
        n_bases = len(cls.__bases__)
        if n_bases == 1:
            return cls.__bases__[0]
        elif n_bases > 1:
            logger.warning("more than one base class is NOT RECOMMENDED")
            for base in cls.__bases__:
                if isinstance(base, Algorithm):
                    return base

    @classmethod
    def _register(cls, impl_type, impl_cls):
        if impl_type in cls._impls:
            logger.warning("override an existing implementation")
        cls._impls[impl_type] = impl_cls
"""