import logging
from pprint import pprint
import unittest
from unittest import TestCase

import coloredlogs

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class FooAlgo(metaclass=AbstractAlgorithm):
    pass

class FooAlgo_CPU(FooAlgo):
    _strategy = ImplTypes.CPU_ONLY

    def __init__(self):
        logger.debug("FooAlgo_CPU.__init__")

    def run(self):
        logger.info("run from CPU")

class BarAlgo(metaclass=AbstractAlgorithm):
    def __init__(self):
        logger.debug("BarAlgo.__init__")
        self.hello = 42

    @interface
    def run(self):
        logger.info("ERROR")

class BarAlgo_GPU(BarAlgo):
    _strategy = ImplTypes.GPU

    def run(self):
        logger.info("run from GPU")

def test():
    class BarAlgo_Dist(BarAlgo):
        _strategy = ImplTypes.DISTRIBUTED

        def _run(self):
            logger.info("run from DISTRIBUTED")


foo = FooAlgo(ImplTypes.CPU_ONLY)
bar = BarAlgo(ImplTypes.GPU)

print("[foo]")
pprint(foo._impl)
print(foo.run())

print()

print("[bar]")
pprint(bar._impl)
print(bar.run())

"""
class Test(TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
"""
