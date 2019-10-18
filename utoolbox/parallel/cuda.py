import logging

import cupy as cp
from mako.template import Template

from utoolbox.utils import AttrDict

__all__ = ["RawKernelFile"]

logger = logging.getLogger(__name__)


class RawKernelFile(AttrDict):
    def __init__(self, path, **kwargs):
        with open(path, "r") as fd:
            template = Template(fd.read())
            self._source = template.render(**kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # lazy load kernel definition, will NOT compile until called
            kernel = cp.RawKernel(self._source, key)
            super().__setitem__(key, kernel)
            return kernel
