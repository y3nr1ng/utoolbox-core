"""
Simple utility function to determine which API to use. Though OpenCL is not
actively used in this toolbox after 2018/08, initialization functions are left
for future references.
"""
try:
    import pycuda
    from .cuda import *

    # cuda driver api needs explicit initialization
    import pycuda.driver
    pycuda.driver.init()
except ImportError:
    import pyopenl
    from .opencl import *
