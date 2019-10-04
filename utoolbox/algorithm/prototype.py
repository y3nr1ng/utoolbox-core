import enum


__all__ = ["ImplTypes", "AbstractAlgorithm", "interface"]


class ImplTypes(enum.IntFlag):
    """Types of implementations."""

    CPU_ONLY = 1  #: only requires CPU resource
    GPU = 2  #: in addition to CPU, this implementation requires GPU
    DISTRIBUTED = 4  #: algorithm can function across worker nodes


INTERFACE_FLAG = "_interface"


def interface(func):
    """
    This decorator sets a label that marks this function to be an requirement 
    in all subclasses.
    """
    setattr(func, INTERFACE_FLAG, True)
    return func


class AlgorithmFactory(type):
    """
    Support class that modify the generated classes to be a factory.
    """

    def __call__(cls, impl_type, *args, **kwargs):
        try:
            subcls = cls._impl[impl_type]
        except KeyError:
            raise KeyError("invalid implementation")
        obj = object.__new__(subcls)
        obj.__init__(*args, **kwargs)
        return obj


class AbstractAlgorithm(type):
    """
    An algorithm container factory with necessary hooks for implementation
    detection.
    """

    def __new__(cls, name, bases, dct):
        # list of concrete implementations
        dct["_impl"] = dict()

        # required interface functions
        int_funcs = []
        for func_name, func in dct.items():
            if hasattr(func, INTERFACE_FLAG):
                int_funcs.append(func_name)
        dct["_int_funcs"] = int_funcs

        new_cls = type.__new__(AlgorithmFactory, name, bases, dct)

        def __init_subclass__(cls):
            """Register an implementation to its target algorithm."""
            # confirm interface functions exist
            for func_name in super(cls, cls)._int_funcs:
                if func_name not in cls.__dict__:
                    raise RuntimeError(
                        'incomplete interface, missing "{}"'.format(func_name)
                    )
            # register this subclass
            try:
                super(cls, cls)._impl[cls._strategy] = cls
            except AttributeError:
                raise AttributeError("unable to determine implementation type")

        setattr(new_cls, "__init_subclass__", classmethod(__init_subclass__))

        return new_cls

