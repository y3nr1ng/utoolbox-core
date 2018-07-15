from collections import defaultdict
import json
import logging
import os
import re

import xarray as xr

logger = logging.getLogger(__name__)

def stocktaking(root_dir):
    """Scan through provided folder and stocktake all available SPIM datasets.
    """
    file_list = os.listdir(root_dir)

    datasets_file_list = defaultdict(lambda : defaultdict(list))
    for file_name in file_list:
        # ignore non-image data
        if not file_name.endswith(".tif"):
            continue

        split_name = file_name.split('_', 1)

        # dual camera?
        dataset_name = split_name[0]
        residue = split_name[1]
        if residue.startswith("Cam"):
            dataset_name += "_" + residue.split('_', 1)[0]

        # ignore settings
        if residue.endswith("Settings.txt"):
            continue

        wavelength = int(re.match(".*_(?P<w>\d+)nm_.*", residue).group("w"))
        datasets_file_list[dataset_name][wavelength].append(file_name)

    # ascending order
    for _, channels in datasets_file_list.items():
        for _, file_list in channels.items():
            file_list.sort(key=
                lambda x: int(re.match(".*_stack(?P<n>\d+)_.*", x).group("n")))

    return datasets_file_list

def open_volume(file_path):
    import imageio

    array = imageio.volread(file_path)
    array = xr.DataArray(data=array, dims=('z', 'y', 'x'))
    return array

def open_dataset(root_dir, rescan=False, parallel=True):
    import dask

    inventory_file = os.path.join(root_dir, "inventory.json")
    #TODO compare for inventory updates using hash
    if not os.path.exists(inventory_file) or rescan:
        datasets_file_list = stocktaking(root_dir)
        with open(inventory_file, "w") as fd:
            json.dump(datasets_file_list, fd, indent=2, sort_keys=True)
    else:
        logger.info("found an inventory file")
        with open(inventory_file, "r") as fd:
            datasets_file_list = json.load(fd)

    _open_volume = dask.delayed(open_volume)
    #_open_volume = open_volume

    for dataset, channels in datasets_file_list.items():
        for channel, file_list in channels.items():
            datasets_file_list[dataset][channel] = [
                _open_volume(os.path.join(root_dir, file_name)) for file_name in file_list]

    return datasets_file_list
