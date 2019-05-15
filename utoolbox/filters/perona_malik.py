"""
Perona-Malik anisotropic smoothing filter [Sergeevich]_.

.. [Sergeevich] Employing the Perona - Malik anisotropic filter for the problem of landing site detection: https://github.com/Galarius/pm-ocl
"""
import logging

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

###
# region: Kernel definitions
###

cu_file = os.path.join(os.path.dirname(__file__), 'perona_malik.cu')

with open(cu_file, 'r') as fd:
    #TODO adjust template
    source = fd.read()

###
# endregion
###

class PeronaMalik(object):
    def __init__(self):
        pass
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
        
    def run_once(self):
        pass

    def run(self):
        pass
    
def perona_malik():
    """Helper function that wraps :class:`.PeronaMalk` for one-off use."""
    pass