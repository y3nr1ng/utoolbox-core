import logging
import os
import re

import click
import pandas as pd

logger = logging.getLogger(__name__)

def zpatch(src, zint, inplace=True, pattern="layer_(\d+)", z_col_header="z [nm]"):
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
@click.option('--no-inplace', 'inplace', is_flag=True, default=True)
@click.argument('src', type=click.Path(exists=True))
@click.argument('zint', type=int)
def zpatch_cli(src, zint, inplace=True):
    """
    Adding Z column filled with ZINT for generated particle list SRC, which can
    be either a directory full of CSV files or a path to a CSV file.
    """
    if os.path.isdir(src):
        fnames = os.listdir(src)
        logger.info("processing a directory, {} files".format(len(fnames)))
        for fname in fnames:
            fname = os.path.join(src, fname)
            try:
                zpatch(fname, zint, inplace)
            except Exception as e:
                # ignore files that cannot process
                logger.warning(e)
    else:
        zpatch(fname, zint, inplace)

if __name__ == '__main__':
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])

    try:
        zpatch_cli()
    except Exception as e:
        logger.error(e)
