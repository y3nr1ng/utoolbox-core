import logging
import os

import click
import coloredlogs

from utoolbox.data.datastore import FolderDatastore
from utoolbox.data.io.amira import AmiraPointCloud


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
            parent, dst = os.path.split(src)
            dst = os.path.join(parent, f"{dst}_csv")

        def write_func(uri, df):
            df.to_csv(uri, index=False)

        ds_in = FolderDatastore(src, read_func=AmiraPointCloud)
        ds_out = FolderDatastore(dst, write_func=write_func)

        logger.info(f"{len(ds_in)} file(s) to process")
        for name, pc in ds_in.items():
            print(f".. {name}")
            ds_out[name] = pc


if __name__ == "__main__":
    main()
