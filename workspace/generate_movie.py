import logging
import os

import coloredlogs
import ffmpeg
import imageio
import numpy as np

from utoolbox.container.datastore import FolderDatastore

logging.getLogger("tifffile").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

root = "~/nas/hive_archive_ytliu/20190528_Cornea/G"

ds = FolderDatastore(
    root, read_func=imageio.imread, extensions=["tif"]
)

# dummy read
ny, nx = next(iter(ds.values())).max(axis=0).shape

# expand path
root = os.path.abspath(root)
parent, basename = os.path.dirname(root), os.path.basename(root)
out_path = os.path.join(parent, "{}.mp4".format(basename))

# invoke ffmpeg
ffmpeg_process = (
    ffmpeg.input(
        "pipe:", format="rawvideo", pix_fmt="gray", s="{}x{}".format(nx, ny)
    )
    .output(out_path, pix_fmt="gray")
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

u8_max = np.iinfo(np.uint8).max
for key, im in ds.items():
    logger.info(key)

    # in
    data = im.astype(np.float32)

    # normalize
    m, M = data.min(), data.max()
    data = (data - m) / (M - m)
    data *= u8_max

    print(data.dtype)

    # out
    data = data.astype(np.uint8)

    ffmpeg_process.stdin.write(data.tobytes())

ffmpeg_process.stdin.close()
ffmpeg_process.wait()