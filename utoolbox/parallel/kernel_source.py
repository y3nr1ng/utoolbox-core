import cupy as cp
from mako.Template import Template

from .attrdict import AttrDict

__all__ = ["RawKernelFile"]


class RawKernelFile(AttrDict):
    def __init__(self, path, **kwargs):
        with open(path, "r") as fd:
            template = Template(fd.read())
            self._source = template.render(**kwargs)

    def __setattr__(self, key, value):
        raise RuntimeError("read-only")

    def __setitem__(self, key, value):
        raise RuntimeError("read-only")

    def __getitem__(self, key):
        try:
            return super()[key]
        except KeyError:
            # lazy load kernel definition, will NOT compile until called
            kernel = cp.RawKernel(self._source, key)
            super()[key] = kernel
            return kernel
