import logging
import os
import re

import click
import coloredlogs
import pandas as pd

import utoolbox.latticescope as llsm
from utoolbox.parallel.gpu import create_some_context
from utoolbox.transform import DeskewTransform

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def deskew(src, zint, inplace=True, pattern="layer_(\d+)", z_col_header="z [nm]"):
    match = re.search(pattern, src)
    if not match:
        raise ValueError("unknown filename '{}'".format(src))
    z = match.group(1)
    try:
        z = int(z)
    except ValueError:
        fname = os.path.basename(src)
        raise ValueError("unable to extract Z index from filename '{}'".format(fname))

    df = pd.read_csv(src, header=0)
    df[z_col_header] = (z-1) * zint

    if not inplace:
        os.rename(src, "{}.old".format(src))
    df.to_csv(src, float_format='%g', index=False)

@click.command()
@click.argument('src_dir', type=click.Path(exists=True))
@click.option('--rotate', 'rotate', is_flag=True, default=False)
@click.option('--dst_dir', 'dst_dir', type=click.Path(exists=False))
def main(src_dir, rotate):
    """
    TBA
    """
    #TODO wrap imageio.volread
    def _zpatch(*args, **kwargs):
        try:
            zpatch(*args, **kwargs)
        except Exception as e:
            logger.error(e)

    ctx = create_some_context()
    ctx.push()

    transform = DeskewTransform(spacing, 32.8, rotate=True)

    ds = llsm.Dataset(src_dir, refactor=False)
    for I_in in ds.datastore:
        I_out = transform(I_in)

    cuda.Context.pop()

if __name__ == '__main__':
    main()
