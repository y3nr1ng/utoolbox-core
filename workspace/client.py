from ftplib import FTP
from functools import partial
import glob
from itertools import zip_longest
import logging
from multiprocessing import Pool
import os

import click

logger = logging.getLogger(__name__)


@click.command()
@click.argument("server")
@click.argument("root")
@click.option("-k", "--keyword", type=str, default="")
@click.option("-r", "--recursive", is_flag=True, default=True)
@click.option("-n", "--workers", "max_workers", type=int, default=20)
def main(server, root, keyword, recursive, max_workers):
    # absolute path
    root = os.path.abspath(root)

    pattern = "*"
    if keyword:
        pattern += keyword + "*"
    if recursive:
        pattern = f"**/{pattern}"
    logger.debug(f'scan pattern "{pattern}"')
    files = [
        os.path.join(root, f)
        for f in glob.glob(os.path.join(root, pattern), recursive=recursive)
    ]
    logger.info(f"{len(files)} file(s) to upload")

    passcode = input("\n Please enter your one-time passcode: ")
    print()

    n_files = len(files)
    chunk_size = n_files // max_workers + 1
    chunks = list(
        zip_longest(
            *[files[i : i + chunk_size] for i in range(0, n_files, chunk_size)],
            fillvalue=None,
        )
    )

    # construct function
    func = partial(upload_files, server, passcode)
    with Pool(max_workers) as pool:
        pool.map(func, chunks)


def upload_files(server, passcode, files):
    ftp = FTP()

    ftp.connect(server, 2121)
    ftp.login("user", passcode)

    for _file in files:
        if _file is None:
            # run into fillers, early stop
            break
        with open(_file, "rb") as fd:
            _file = os.path.basename(_file)
            print(f".. {_file}")
            ftp.storbinary(f"STOR {_file}", fd, 16384)

    ftp.close()


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
