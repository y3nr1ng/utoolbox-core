# pylint: disable=W0612

import logging
from pprint import pprint

import coloredlogs
import pytest
from pytest import fixture

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

@fixture
def algo():
    class Foo(metaclass=AbstractAlgorithm):
        @interface
        def run(self):
            pass
    return Foo

def test_correct_implementation(algo):
    class Foo_CPU(algo): 
        _strategy = ImplTypes.CPU_ONLY

        def run(self):
            print("Foo, CPU")

def test_incomplete_interface(algo):
    with pytest.raises(RuntimeError) as _:
        class Foo_CPU(algo):
            _strategy = ImplTypes.CPU_ONLY

def test_undefined_implementation(algo):
    class Foo_CPU(algo):
        _strategy = ImplTypes.CPU_ONLY

        def run(self):
            print("Foo, CPU")
    
    with pytest.raises(KeyError) as _:
        foo = algo(ImplTypes.GPU)
        
def test_multiple_implementation(algo):
    class Foo_CPU(algo):
        _strategy = ImplTypes.CPU_ONLY

        def run(self):
            print("Foo, CPU")
    
    class Foo_GPU(algo):
        _strategy = ImplTypes.GPU

        def run(self):
            print("Foo, GPU")
    
    assert len(algo._impl) == 2
    assert (ImplTypes.CPU_ONLY in algo._impl) and (ImplTypes.GPU in algo._impl)

"""
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

class Test(TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
"""
