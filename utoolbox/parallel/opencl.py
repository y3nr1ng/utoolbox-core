import logging
from collections import namedtuple
import operator

import pyopencl as cl

__all__ = [
    'list_devices',
    'list_device_info',
    'create_some_context',
    'parse_cq'
]

logger = logging.getLogger(__name__)

_device_types = {
    'cpu': cl.device_type.CPU,
    'gpu': cl.device_type.GPU
}

_vendors = ('intel', 'nvidia', 'amd')

def list_devices(dev_type=None, vendor=None):
    dev_type = _device_types.get(dev_type.lower(), cl.device_type.ALL)
    if vendor:
        vendor = vendor.lower()
        if vendor not in _vendors:
            raise ValueError("unknown vendor")
        else:
            is_wanted = lambda d: d.get_info(cl.device_info.VENDOR).lower() == vendor
    else:
        is_wanted = lambda d: True

    dev_list = []
    for _platform in cl.get_platforms():
        for _device in _platform.get_devices(device_type=dev_type):
            if is_wanted(_device):
                dev_list.append(_device)
    return dev_list

def list_device_info(device, friendly=False):
    di = cl.device_info
    flags = {
        'available': di.AVAILABLE,
        'type': di.TYPE,
        'platform': di.PLATFORM,
        'vendor': di.VENDOR,
        'global_mem': di.GLOBAL_MEM_SIZE,
        'image_support': di.IMAGE_SUPPORT,
        'max_const_size': di.MAX_CONSTANT_BUFFER_SIZE,
        'max_alloc_size': di.MAX_MEM_ALLOC_SIZE
    }
    for k, v in flags.items():
        v = device.get_info(v)
        if friendly and k in ('global_mem', 'max_const_size', 'max_alloc_size'):
            import humanfriendly
            v = humanfriendly.format_size(v, binary=True)
        flags[k] = v
    return flags

def create_some_context(dev_type=None, vendor=None):
    devices = list_devices(dev_type, vendor)
    selected_devices = None
    if not devices:
        raise RuntimeError("unable to find an usable device")
    elif len(devices) == 1:
        selected_devices = devices[0]
    else:
        print("Choose device(s):")
        for i, device in enumerate(devices):
            print("[{}] {}".format(i, device.get_info(cl.device_info.NAME)))
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
    def create_context(device):
        platform = device.get_info(cl.device_info.PLATFORM)
        return cl.Context(
            devices=[device],
            properties=[(cl.context_properties.PLATFORM, platform)]
        )
    if isinstance(selected_devices, cl.Device):
        return create_context(selected_devices)
    else:
        logger.debug("{} devices selected, "
                     "multiple context returned".format(len(selected_devices)))
        return list(create_context(device) for device in selected_devices)

def parse_cq(cq):
    """Re-interpret context or queue into both."""
    if isinstance(cq, cl.CommandQueue):
        return cq.context, cq
    elif isinstance(cq, cl.Context):
        logger.info("requesting command queue from context")
        return cq, cl.CommandQueue(cq)
    else:
        raise TypeError("cq may be a queue or a context, not '%s'".format(type(cq)))
