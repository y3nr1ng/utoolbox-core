import logging
import os
from pprint import pprint

import click
import coloredlogs
import h5py
import numpy as np

from utoolbox.data.dataset import MicroManagerDataset, BigDataViewerXML
from utoolbox.utils.decorator import timeit

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


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

        # downsample range
        ranges = tuple(slice(None, None, step) for step in subsample)

        # new dataset
        path = f"t{0:05d}/s{ss:02d}/{i}"
        logger.debug(f".. > {path}")
        group = handle.create_group(path)
        group.create_dataset(
            "cells",
            data=data[ranges],
            chunks=chunk,
            scaleoffset=0,
            compression=compression,
            shuffle=True,
        )

    # actual write back
    logger.debug(f".. > flushing..")
    handle.flush()


def find_voxel_size(metadata):
    # X
    dx = metadata["PixelSize_um"]

    # Y
    r = metadata["PixelAspect"]
    dy = r * dx

    # Z
    dz = metadata["z-step_um"]
    dz = abs(dz)

    return dz, dy, dx


@click.command()
@click.argument("src_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "dst_dir",
    help="Output directory, default directory appends hdf5.",
)
@click.option(
    "-d",
    "--dry",
    "dry_run",
    is_flag=True,
    default=False,
    help="Dry run, generate XML only.",
)
def main(src_path, dst_dir=None, dry_run=False):
    """
    Convert Micro-Manager dataset to BigDataViewer complient XML/HDF5 format.

    Args:
        src_path (str): path to the MM dataset
        dst_path (str, optional): where to save the BDV dataset
        dry_run (bool, optinal): save XML only
    """
    dataset = MicroManagerDataset(src_path, force_stack=True)

    pprint(dataset.metadata)

    if dst_dir is None:
        dst_dir = f"{src_path}_hdf5"
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass
    h5_path = os.path.join(dst_dir, f"dataset.h5")

    xml = BigDataViewerXML(h5_path)
    voxel_size = find_voxel_size(dataset.metadata)

    if dry_run:
        for channel, datastore in dataset.items():
            with datastore as source:
                for key, data in source.items():
                    ss = xml.add_view(channel, data, name=key, voxel_size=voxel_size)
                    logger.info(f".. [{ss}] {key}")
    else:
        with h5py.File(h5_path, "a") as h:
            # Why declare this?
            h["__DATA_TYPES__/Enum_Boolean"] = np.dtype("bool")

            for channel, datastore in dataset.items():
                with datastore as source:
                    for key, data in source.items():
                        ss = xml.add_view(
                            channel, data, name=key, voxel_size=voxel_size
                        )
                        logger.info(f".. [{ss}] {key}")
                        save_to_hdf(
                            h,
                            ss,
                            data,
                            [(1, 2, 2), (2, 4, 4), (4, 8, 8)],
                            [(4, 32, 32), (16, 16, 16), (16, 16, 16)],
                        )
    xml.serialize()


if __name__ == "__main__":
    main(
        "Z:/charm/Clarity_Brain/perfus_Lectin594_poststain_lectin647_2x0bj_1",
        "E:/lectin594_poststain_lectin647_2x0bj",
        dry_run=True,
    )

