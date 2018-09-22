from pprint import pprint

from utoolbox.parallel import gpu

for device in gpu.list_devices():
    pprint(gpu.list_device_info(device, friendly=True))
