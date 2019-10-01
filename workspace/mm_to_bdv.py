import logging
import os
from pprint import pprint
from xml.etree import ElementTree as ET

import coloredlogs
import h5py
import numpy as np

from utoolbox.data.dataset import MicroManagerDataset
from utoolbox.utils.decorator import timeit

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


def find_voxel_shape(metadata, dx=None, dy=None, dz=None):
    if not dx:
        dx = metadata["PixelSize_um"]
    if not dy:
        r = metadata["PixelAspect"]
        dy = r * dx
    if not dz:
        dz = metadata["z-step_um"]
    return (abs(dz), dy, dx)


def generate_xml(hdf_path, dataset):
    # build XML header
    root = ET.Element("SpimData")
    root.set("version", "0.2")
    # set base path to current folder
    base_path = ET.SubElement(root, "BasePath")
    base_path.set("type", "relative")
    base_path.text = "."

    sequence = ET.SubElement(root, "SequenceDescription")

    # set hdf filename
    loader = ET.SubElement(sequence, "ImageLoader")
    loader.set("format", "bdv.hdf5")
    loader_fname = ET.SubElement(loader, "hdf5")
    loader_fname.set("type", "relative")
    loader_fname.text = os.path.basename(hdf_path)

    # contents
    setups = ET.SubElement(sequence, "ViewSetups")
    attributes = ET.SubElement(setups, "Attributes")
    attributes.set("name", "channel")
    regs = ET.SubElement(root, "ViewRegistrations")

    # extract voxel info from metadata
    voxel_shape = find_voxel_shape(dataset.metadata, dx=1)

    # loop over the channels
    ss = 0
    for c, (channel, datastore) in enumerate(dataset.items()):
        # save channel description
        attrs_ch = ET.SubElement(attributes, "Channel")
        ET.SubElement(attrs_ch, "id").text = str(c)
        ET.SubElement(attrs_ch, "name").text = channel

        with datastore as source:
            data_shape = source._buffer.shape

            for tile in datastore.keys():
                setup = ET.SubElement(setups, "ViewSetup")

                # setup description
                ET.SubElement(setup, "id").text = str(ss)
                ET.SubElement(setup, "name").text = tile
                ET.SubElement(setup, "size").text = " ".join(
                    str(s) for s in reversed(data_shape)
                )

                setup_attrs = ET.SubElement(setup, "attributes")
                ET.SubElement(setup_attrs, "channel").text = str(c)

                # voxel description
                setup_voxel = ET.SubElement(setup, "voxelSize")
                ET.SubElement(setup_voxel, "unit").text = "micron"
                ET.SubElement(setup_voxel, "size").text = " ".join(
                    str(s) for s in reversed(voxel_shape)
                )
                # affine transformation by voxel
                reg = ET.SubElement(regs, "ViewRegistration")
                reg.set("timepoint", str(0))
                reg.set("setup", str(ss))
                trans = ET.SubElement(reg, "ViewTransform")
                trans.set("type", "affine")
                ET.SubElement(
                    trans, "affine"
                ).text = "{:.4f} 0.0 0.0 0.0 0.0 {:.4f} 0.0 0.0 0.0 0.0 {:.4f} 0.0".format(
                    *(voxel_shape[::-1])
                )

                # next setup
                ss += 1

    # write dummy timepoint
    tp = ET.SubElement(sequence, "Timepoints")
    tp.set("type", "range")
    ET.SubElement(tp, "first").text = str(0)
    ET.SubElement(tp, "last").text = str(0)

    # write back
    tree = ET.ElementTree(root)
    return tree


@timeit
def save_to_hdf(
    handle,
    ss,
    data,
    subsamples=[(1, 1, 1), (2, 4, 4)],
    chunks=[(4, 32, 32), (16, 16, 16)],
    compression="gzip",
):
    """
    Args:
        handle (File): HDF file session
        ss (int): view setup ID
        stack (np.ndarray): the data to save
        subsample (list of tuples): subsample specifications
        chunks (list of tuples): chunk size per subsample level
    """
    # resolutions and subdivisions, resizable axis to allow additional levels
    group = handle.create_group(f"s{ss:02d}")
    # .. subsample
    _subsamples = np.array(subsamples)
    group.create_dataset(
        "resolutions",
        data=np.fliplr(_subsamples),
        dtype="<f8",
        chunks=_subsamples.shape,
        maxshape=(None, None),
    )
    # .. chunks
    _chunks = np.array(chunks)
    group.create_dataset(
        "subdivisions",
        data=np.fliplr(_chunks),
        dtype="<i4",
        chunks=_chunks.shape,
        maxshape=(None, None),
    )

    # NOTE BDV cannot handle uint16
    data = data.astype(np.int16)

    for i, (subsample, chunk) in enumerate(zip(subsamples, chunks)):
        logger.debug(f".. > subsample: {subsample}, chunk: {chunk}")

        path = f"t{0:05d}/s{ss:02d}/{i}"
        logger.debug(f".. > {path}")
        group = handle.create_group(path)
        ranges = tuple(slice(None, None, step) for step in subsample)
        group.create_dataset(
            "cells",
            data=data[ranges],
            chunks=chunk,
            scaleoffset=0,
            compression=compression,
        )


def main(src_path, dst_dir=None, dry_run=False):
    """
    Args:
        src_path (str): path to the MM dataset
        dst_path (str, optional): where to save the BDV dataset
        dry_run (bool, optinal): save XML only
    """
    dataset = MicroManagerDataset(src_path, force_stack=True)

    if dst_dir is None:
        dst_dir = f"{src_path}_hdf5"
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass
    hdf_path = os.path.join(dst_dir, f"dataset.h5")
    xml_path = os.path.join(dst_dir, f"dataset.xml")

    # extract voxel info
    pprint(dataset.metadata)

    if not dry_run:
        with h5py.File(hdf_path, "w") as h:
            # Why declare this?
            h["__DATA_TYPES__/Enum_Boolean"] = np.dtype("bool")

            ss = 0
            for channel, datastore in dataset.items():
                with datastore as source:
                    for key, value in source.items():
                        logger.info(f".. [{ss}] {key}")
                        save_to_hdf(
                            h,
                            ss,
                            value,
                            [(2, 2, 2), (4, 8, 8)],
                            [(16, 16, 16), (16, 16, 16)],
                        )
                        ss += 1

    # save the xml
    tree = generate_xml(hdf_path, dataset)
    tree.write(xml_path)
    logger.info(f'XML saved to "{xml_path}"')


if __name__ == "__main__":
    # main("E:/brain_in_XClarity_4")
    main(
        "Z:/charm/Clarity_Brain/perfus_Lectin594_poststain_lectin647_2x0bj_1",
        "E:/lectin594_poststain_lectin647_2x0bj",
        dry_run=False,
    )

