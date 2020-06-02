from functools import reduce
import logging
import operator
import os
from pprint import pprint

import click
import coloredlogs
import h5py
import numpy as np

from utoolbox.data.dataset import (
    MicroManagerV1Dataset,
    MicroManagerV2Dataset,
    BigDataViewerXML,
)
from utoolbox.data.dataset.error import DatasetError
from utoolbox.util.decorator import timeit

coloredlogs.install(
    level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


@timeit
def save_to_hdf(
    handle,
    ss,
    data,
    downsamples=[(1, 1, 1), (2, 4, 4)],
    chunks=(64, 128, 128),
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
    _downsamples = np.array(downsamples)
    group.create_dataset(
        "resolutions",
        data=np.fliplr(_downsamples),
        dtype="<f8",
        chunks=_downsamples.shape,
        maxshape=(None, 3),
    )
    print(np.fliplr(_downsamples).shape)

    # .. chunks
    # expand chunk setup if necessary
    if isinstance(chunks[0], int):
        chunks = (chunks,)
        chunks *= len(downsamples)
    _chunks = np.array(chunks)
    group.create_dataset(
        "subdivisions",
        data=np.fliplr(_chunks),
        dtype="<i4",
        chunks=_chunks.shape,
        maxshape=(None, 3),
    )

    # NOTE BDV cannot handle uint16
    data = data.astype(np.int16)

    for i, (downsample, chunk) in enumerate(zip(downsamples, chunks)):
        logger.debug(f".. > subsample: {downsample}, chunk: {chunk}")

        # downsample range
        ranges = tuple(slice(None, None, step) for step in downsample)

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


def find_voxel_size(info):
    return (info.z_step,) + info.pixel_size


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
@click.option(
    "-s",
    "--downsample",
    "downsamples",
    nargs=3,
    type=int,
    multiple=True,
    help='downsample ratio along "X Y Z" axis',
)
def main(src_path, dst_dir=None, dry_run=False, downsamples=[(1, 1, 1), (2, 2, 2)]):
    """
    Convert Micro-Manager dataset to BigDataViewer complient XML/HDF5 format.
    \f

    Args:
        src_path (str): path to the MM dataset
        dst_path (str, optional): where to save the BDV dataset
        dry_run (bool, optinal): save XML only
        downsamples (tuple of int, optional): downsample ratio along (X, Y, Z) axis
    """
    for klass in (MicroManagerV1Dataset, MicroManagerV2Dataset):
        try:
            dataset = klass(src_path, force_stack=True)
            break
        except DatasetError:
            pass
    else:
        raise RuntimeError("unknown dataset format")

    print("== info ==")
    for key, value in dataset.info.items():
        pprint(f"{key}: {value}")
    print()

    if dst_dir is None:
        dst_dir = f"{src_path}_hdf5"
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass
    h5_path = os.path.join(dst_dir, f"dataset.h5")

    xml = BigDataViewerXML(h5_path)
    voxel_size = find_voxel_size(dataset.info)

    if dry_run:
        for channel, datastore in dataset.items():
            with datastore as source:
                for i_tile, key in enumerate(source.keys()):
                    ss = xml.add_view(
                        channel,
                        source._buffer,
                        name=key,
                        voxel_size=voxel_size,
                        tile=i_tile,
                    )
                    logger.info(f".. [{ss}] {key}")
    else:
        # ensure downsamples is wrapped
        if isinstance(downsamples[0], int):
            downsamples = [downsamples]
        # reverse downsampling ratio
        downsamples = [tuple(reversed(s)) for s in downsamples]

        # estimate cache size
        chunk_size = (64, 64, 64)
        max_slots = reduce(operator.mul, (1024 // c for c in chunk_size[1:]), 1)
        rdcc_nbytes = reduce(operator.mul, chunk_size, 1) * max_slots * 2
        logger.info(f"cache size: {rdcc_nbytes} bytes")

        with h5py.File(
            h5_path, "w", rdcc_nbytes=rdcc_nbytes, rdcc_nslots=max_slots
        ) as h:
            # Why declare this?
            h["__DATA_TYPES__/Enum_Boolean"] = np.dtype("bool")

            for channel, datastore in dataset.items():
                with datastore as source:
                    for i_tile, (key, data) in enumerate(source.items()):
                        ss = xml.add_view(
                            channel, data, name=key, voxel_size=voxel_size, tile=i_tile
                        )
                        logger.info(f".. [{ss}] {key}")
                        save_to_hdf(h, ss, data, downsamples, chunk_size)
    xml.serialize()


if __name__ == "__main__":
    main("Z:/charm/20181009_ExM_4x_hippocampus", dry_run=True)

