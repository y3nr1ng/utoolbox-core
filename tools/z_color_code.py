# pylint: disable=E1120
import io
import logging
from math import radians, sin, cos, ceil
import os

import click
import coloredlogs
import imageio
import numpy as np
from tqdm import tqdm

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of stdout.
    
    Reference
    ---------
    https://github.com/tqdm/tqdm/issues/313#issuecomment-267959111
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

tqdm_log = TqdmToLogger(logger, level=logging.INFO)

def colored_by_depth_spl(I, lut, lateral, axial, rad):
    imin, imax = np.amin(I).astype(np.float32), np.amax(I).astype(np.float32)
    logger.info("input range [{}, {}]".format(imin, imax))

    nz, ny, nx = I.shape
    Dx = np.arange(nx, dtype=np.float32)
    J = np.empty((nz, ny, nx, 3), dtype=lut.dtype)
    for iz, Iz in tqdm(enumerate(I), total=nz, file=tqdm_log):
        # current shifts
        D = np.repeat(Dx[None, :], ny, axis=0)
        D -= np.float32(axial*cos(rad)) * iz

        # projected z
        D *= np.float32(sin(rad))

        Dp = lut(D)
        Iz = (Iz.astype(np.float32)-imin)/(imax-imin)
        for ic in range(3):
            J[iz, :, :, ic] = Dp[..., ic] * Iz
    
    return J

def colored_by_depth_obj(I, lut):
    imin, imax = np.amin(I).astype(np.float32), np.amax(I).astype(np.float32)
    logger.info("input range [{}, {}]".format(imin, imax))

    

    I = I.astype(np.float32)
    imin, imax = np.min(I), np.max(I)
    In = (I-imin)/(imax-imin)
    logger.debug("input intensity normalized")

    nz, _, _ = I.shape
    J = np.empty(I.shape + (3, ), dtype=lut.dtype)
    for iz, _I in tqdm(enumerate(In), total=nz, file=tqdm_log):
        for i in range(3):
            J[iz, :, :, i] = (lut[iz, i] * _I).astype(J.dtype)
    return J

def colored_by_ortho(I, lut):
    pass
        
def colored_by_intensity(I, lut):
    J = np.empty(I.shape + (3, ), dtype=lut.dtype)
    for i in range(3):
        J[..., i] = lut[I, i]
    return J

def create_lookup_function(cm, scale):
    smin, smax = scale

    xp = np.linspace(smin, smax, cm.shape[0])
    def lookup(I):
        # clip the input array to specified range for interpolation
        np.clip(I, smin, smax, out=I)

        # linearize to ease the processing
        shape = I.shape
        If = np.ravel(I)
        J = np.empty((I.size, 3), dtype=cm.dtype)
        for i in range(3):
            J[:, i] = np.interp(If, xp, cm[:, i])

        return np.reshape(J, shape + (3, ))
    lookup.dtype = cm.dtype

    return lookup

@click.command()
@click.option('-t', '--type', 'scan_type', 
              type=click.Choice(['spl', 'obj', 'ortho']), default='obj', 
              help="Provided scan type, sample scan (spl), objective scan (obj) or orthogonal view (ortho), default is obj.")
@click.option('-l', '--lateral', type=float, default=.102,
              help='Lateral resolution in um, default is 0.102')
@click.option('-a', '--axial', type=float, default=.15, 
              help='Axial resolution in um, default is 0.15')
@click.option('--angle', type=float, default=32.8,
              help='Coverslip rotation angle in degrees, default is 32.8')
@click.option('-s', '--suffix', type=str, default='colored', 
              help="Suffix for output file name, default is 'colored'")
@click.argument('image', type=click.Path(exists=True))
@click.argument('colormap', type=click.Path(exists=True))
def main(image, colormap, scan_type, lateral, axial, angle, suffix):
    # load image
    logger.info("loading \"{}\"".format(image))
    I = imageio.volread(image)
    logger.info("image shape {}".format(I.shape))

    # load colormap
    cm = imageio.imread(colormap)
    cm = cm[0, :, :]
    logger.info("colormap contains {} steps".format(cm.shape[0]))

    nz, ny, nx = I.shape
    rad = radians(angle)
    if scan_type == 'spl':
        scale = (np.float32(0.), np.float32(nx*sin(rad)))
        convert = lambda x, y: colored_by_depth_spl(x, y, lateral, axial, rad)
    elif scan_type == 'obj':
        raise NotImplementedError
    elif scan_type == 'ortho':
        raise NotImplementedError
    lookup = create_lookup_function(cm, scale)

    J = convert(I, lookup)

    # apply suffix
    fn, ext = os.path.splitext(image)
    for iz in tqdm(range(nz), total=nz, file=tqdm_log):
        imageio.imwrite(
            '{}_{:03d}_{}{}'.format(fn, iz, suffix, ext), 
            J[iz, ...]
        )

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        logger.error(str(ex))
