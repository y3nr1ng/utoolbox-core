import logging
import os

import click
import imageio
import numpy as np

from utoolbox.analysis import PSFAverage

__all__ = ["analyze_psf"]

logger = logging.getLogger(__name__)


@click.command(short_help="analyze PSF info from a beads-coated slide stack")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--ratio",
    "-r",
    type=float,
    default=2,
    show_default=True,
    help="S.D. ratio for peak detection",
)
@click.option(
    "--lateral",
    "-l",
    type=int,
    default=33,
    show_default=True,
    help="lateral bounding box size",
)
@click.option(
    "--axial",
    "-a",
    type=int,
    default=65,
    show_default=True,
    help="axial bounding box size",
)
@click.option("--preview", is_flag=True)
@click.option("--save-roi", is_flag=True)
@click.option("--no-fix-negative", is_flag=True)
@click.pass_context
def analyze_psf(ctx, path, ratio, lateral, axial, preview, save_roi, no_fix_negative):
    path = os.path.abspath(path)

    data = imageio.volread(path)
    if not no_fix_negative:
        vmin = np.min(data)
        if vmin < 0:
            data -= vmin

    analyzer = PSFAverage((axial, lateral, lateral), ratio)
    rois, table = analyzer(data)
    
    out_dir, _ = os.path.splitext(path)
    out_dir = f'{out_dir}_psf'
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        logger.warning(f'output directory "{out_dir}" exists')

    if save_roi:
        logger.info("saving ROIs...")
        for i, roi in enumerate(rois):
            out_path = os.path.join(out_dir, f'psf_{i:03d}.tif')
            imageio.volwrite(out_path, roi)

    if preview:
        fig, ax = plt.subplots()
        plt.ion()
        plt.show()

