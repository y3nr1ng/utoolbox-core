# pylint: disable=no-value-for-parameter
import glob
import logging
import os

import click
import coloredlogs
import imageio
import numpy as np

logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "verbose", count=True)
@click.pass_context
def main(ctx, verbose):
    if verbose == 0:
        verbose = "WARNING"
    elif verbose == 1:
        verbose = "INFO"
    else:
        verbose = "DEBUG"
    coloredlogs.install(
        level=verbose, fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


@main.command("split", short_help="split a stack to files")
@click.argument("src")
def split(src):
    print("split, {}".format(src))


def _split_stack_file(src_fp, dst_dp=None):
    # filename to folder name
    parent, src_fn = os.path.split(src_fp)
    src_fn, src_ext = os.path.splitext(src_fn)
    # redirect output folder
    if not dst_dp:
        dst_dp = parent
    dst_dp = os.path.join(dst_dp, src_fn)

    try:
        os.makedirs(dst_dp)
    except FileExistsError:
        pass

    # split file
    im = imageio.volread(src_fp)
    for iz, _im in enumerate(im):
        dst_fn = "{}_{:03d}{}".format(src_fn, iz, src_ext)
        dst_fp = os.path.join(dst_dp, dst_fn)
        imageio.imwrite(dst_fp, _im)


def _split_stack_dir(src_dp, ext="tif"):
    parent, src_dn = os.path.split(src_dp)
    dst_dp = os.path.join(parent, "{}_split".format(src_dn))

    try:
        os.makedirs(dst_dp)
    except FileExistsError:
        pass

    for fn in glob.iglob(os.path.join(src_dp, "*.{}".format(ext))):
        logger.info('splitting "{}"'.format(fn))
        fn = os.path.join(src_dp, fn)
        _split_stack_file(fn, dst_dp=dst_dp)


@click.command()
@click.argument("root")
def split_stack(root):
    if os.path.isdir(root):
        _split_stack_dir(root)
    else:
        _split_stack_file(root)


if __name__ == "__main__":
    main(obj={})
