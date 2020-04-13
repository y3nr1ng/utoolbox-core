import logging
import os

import click
import coloredlogs

from utoolbox.data.datastore import FolderDatastore
from utoolbox.data.io.amira import AmiraPointCloud

logger = logging.getLogger(__name__)


@click.command()
@click.argument("src", type=click.Path(exists=True))
@click.option("-o", "--output", "dst", type=click.Path())
def main(src, dst=None):
    """
    Convert Amira PointCloud to CSV.

    Args:
        src (str): source path
        dst (str, optional): destination
    """
    if os.path.isfile(src):
        if dst is None:
            name, _ = os.path.splitext(src)
            dst = name + ".csv"
        AmiraPointCloud(src).to_csv(dst, index=False)
    else:
        if dst is None:
            parent, dst = os.path.split(os.path.abspath(src))
            dst = os.path.join(parent, f"{dst}_csv")
            print(dst)

        def write_func(uri, df):
            dst = f"{uri}.csv"
            df.to_csv(dst, index=False)

        ds_in = FolderDatastore(src, read_func=AmiraPointCloud)
        ds_out = FolderDatastore(dst, write_func=write_func)

        logger.info(f"{len(ds_in)} file(s) to process")
        i = 0
        for name in ds_in.keys():
            print(f".. {name}")
            if i < 760 and i < 770:
                i += 1
                continue
            try:
                ds_out[name] = ds_in[name]
            except ValueError as err:
                logger.exception(f"ERROR! {str(err)}")


if __name__ == "__main__":
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
