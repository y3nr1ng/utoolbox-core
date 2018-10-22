import logging
import os
import re

import click
import coloredlogs
import imageio
import pandas as pd
import pycuda.driver as cuda

import utoolbox.latticescope as llsm
from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform import DeskewTransform

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument('src_dir', type=click.Path(exists=True))
@click.argument('angle', type=float)
@click.option('--rotate', 'rotate', is_flag=True, default=False)
@click.option('--dst_dir', 'dst_dir', type=click.Path(exists=False))
def main(src_dir, dst_dir, angle, rotate):
    """
    TBA
    """
    ctx = create_some_context()
    ctx.push()

    transform = DeskewTransform(spacing, angle, rotate=rotate)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    ds = llsm.Dataset(src_dir, refactor=False)
    #TODO: extract filename from datastore
    for I_in in ds.datastore:
        I_out = transform(I_in)
        imageio.volwrite(os.path.join(dst_dir, filename), I_out)

    cuda.Context.pop()

if __name__ == '__main__':
    main()
