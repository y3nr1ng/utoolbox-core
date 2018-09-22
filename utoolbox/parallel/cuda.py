import logging
from collections import namedtuple
import operator

import pycuda.driver as cuda

__all__ = [
    'list_devices',
    'list_device_info',
    'create_some_context'
]

logger = logging.getLogger(__name__)

def list_devices():
    n_dev = cuda.Device.count()
    if n_dev == 0:
        raise RuntimeError("no cuda-enabled device found")
    return [cuda.Device(i_dev) for i_dev in range(n_dev)]

def list_device_info(device, friendly=False):
    da = cuda.device_attribute
    attributes = {
        'compute_capability': device.compute_capability(),
        'total_memory': device.total_memory(),
        'max_const_size': device.get_attribute(da.TOTAL_CONSTANT_MEMORY)
    }
    for k, v in attributes.items():
        if friendly and k in ('total_memory', 'max_const_size'):
            import humanfriendly
            v = humanfriendly.format_size(v, binary=True)
            attributes[k] = v
    return attributes

def create_some_context():
    devices = list_devices()
    if len(devices) == 1:
        selected_devices = devices[0]
    else:
        print("Choose device(s):")
        for i, device in enumerate(devices):
            print("[{}] {}".format(i, device.name()))
        user_input = input("Choice, comma-separated [0]:")
        if not user_input:
            selected_devices = devices[0]
        else:
            try:
                indices = list(int(i) for i in user_input.split(","))
                selected_devices = operator.itemgetter(*indices)(devices)
            except:
                logger.error("invalid choice, use default choice [0]")
                selected_devices = devices[0]
    if isinstance(selected_devices, cuda.driver.Device):
        return selected_devices.make_context()
    else:
        logger.debug("{} devices selected, "
                     "multiple context returned".format(len(selected_devices)))
        contex = list(device.make_context() for device in selected_devices)
        # make the first one default context
        cuda.driver.Context.pop()
        context[0].push()
        return contex
