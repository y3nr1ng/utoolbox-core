from collections import namedtuple
import operator

import pyopencl as cl

DevInfo = namedtuple('DevInfo', ['cl_ver', 'global_mem'])

def list_devices(device_type=cl.device_type.ALL):
    di = cl.device_info

    dev_list = {}
    for platform in cl.get_platforms():
        dev_list[platform] = {}
        for device in platform.get_devices(device_type=device_type):
            dev_list[platform][device] = DevInfo(
                float(device.get_info(di.VERSION).split()[1]),
                device.get_info(di.GLOBAL_MEM_SIZE)
            )
    return dev_list

def pick_n_gpu(n_gpu, sort_by=['global_mem']):
    dev_list = list_devices(device_type=cl.device_type.GPU)
    dev_list = [
        [(platform, device, info) for device, info in devices.items()]
        for platform, devices in dev_list.items()
    ]
    dev_list.sort(key=operator.itemgetter(1, 2))
    #TODO sort by index
    return dev_list[:n_gpu]
