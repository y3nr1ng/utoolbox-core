# pylint: disable=W0612,E1101

import pytest
from pytest import fixture

from utoolbox.container import AbstractAlgorithm, ImplTypes, interface

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

def test_multiple_algorithms(algo):
    class Bar(metaclass=AbstractAlgorithm):
        @interface
        def run(self):
            pass

    class Foo_CPU(algo):
        _strategy = ImplTypes.CPU_ONLY
        def run(self):
            print("Foo, CPU")

    class Bar_CPU(Bar):
        _strategy = ImplTypes.CPU_ONLY
        def run(self):
            print("Bar, CPU")
    
    class Bar_GPU(Bar):
        _strategy = ImplTypes.GPU
        def run(self):
            print("Bar, GPU")
    
    assert (len(algo._impl) == 1) and (len(Bar._impl) == 2)
    assert (ImplTypes.GPU not in algo._impl)
