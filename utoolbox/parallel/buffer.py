from warnings import warn

import pyopencl as cl
import pyopencl.array

from utoolbox.container import AttrDict

pyopencl.array.Array(
    cq, shape, dtype,
    order='C',
     allocator=None,
     data=None,
     offset=0, strides=None, events=None)

class BufferGroup(AttrDict):
    def __init__(self, *args **kwargs):
        super(SafeBuffer, self).__init__(*args, **kwargs)

    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.base_data.release()

class SafeBuffer(AbstractContextManager, cl.array.Array):
    def __init__(self, )
