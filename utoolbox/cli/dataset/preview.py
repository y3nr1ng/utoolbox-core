import logging

import click
import imageio

__all__ = ["preview_dataset"]

logger = logging.getLogger(__name__)


@click.command("preview", short_help="generate preview")
@click.argument("root", type=click.Path(exists=True))
@click.option("--size", type=int, default=1)
@click.option("--format", type=click.Choice(["mp4", "tif"]))
@click.pass_context
def preview_dataset(ctx, root, shrink):
    pass


def dummy():
    ds = FolderDatastore(
        root, read_func=imageio.volread, pattern="*5a_ch0_*", extensions=["tif"]
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

        with Orthogonal(np.asarray(im)) as data_ortho:
            data = data_ortho.xy

        # in
        data = data.astype(np.float32)

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
