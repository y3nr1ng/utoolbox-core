"""
Driver interface for localization microscopy.
"""
import logging
import os
import re

import click
import coloredlogs
import imageio

import utoolbox.latticescope as llsm

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument('src_dir', type=click.Path(exists=True))
@click.argument('angle', type=float)
@click.option('--rotate', '-r', is_flag=True, default=False,
              help='Rotate the data to coverslip orientation.')
@click.option('--dst_dir', type=click.Path(file_okay=False, exists=False),
              help='Alternative output directory.')
def main(src_dir, dst_dir, angle, rotate):
    if not dst_dir:
        suffix = '_deskew' if rotate else '_shear'
        dst_dir = os.path.abspath(src_dir) + suffix
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        logger.info("\"{}\" created".format(dst_dir))

    try:
        ds = llsm.Dataset(src_dir, show_uri=True, refactor=False)
        for fname, I_in in ds.datastore:
            logger.debug("deskew \"{}\"".format(fname))
            I_out = transform(I_in)
            basename = os.path.basename(fname)
            imageio.volwrite(os.path.join(dst_dir, basename), I_out)
    except Exception as e:
        logger.error(e)

if __name__ == '__main__':
    main()
