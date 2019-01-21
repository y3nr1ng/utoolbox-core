import io
import logging
import os

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

def colored_by_depth(I, lut):
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

def colored_by_intensity(I, lut):
    J = np.empty(I.shape + (3, ), dtype=lut.dtype)
    for i in range(3):
        J[..., i] = lut[I, i]
    return J

def create_lut(cm, scale):
    imin, imax = scale # intensity range
    npts = imax-imin+1
    ratio, _ = cm.shape

    cm_s = np.empty((npts, 3), dtype=cm.dtype)
    # intensity lookup
    x = np.linspace(-imin/(imax-imin)*ratio, ratio, npts)
    # cm lookup
    xp = np.arange(0, ratio)

    for i in range(3):
        cm_s[:, i] = np.interp(x, xp, cm[:, i])

    return cm_s

def main():
    # load image
    im_path = 'data/Figure2D_HyperStack_ResliceZ_22To153_Reverse.tif'
    logger.info("loading \"{}\"".format(im_path))
    I = imageio.volread(im_path)
    logger.info("image shape {}".format(I.shape))

    # load colormap
    cm_path = 'data/Figure2D_HyperStack_Spectrum_ResliceZ_TemporalColorCode.tif'
    cm = imageio.imread(cm_path)
    cm = cm[0, :, :]

    nz, _, _ = I.shape
    lut = create_lut(cm, (0, nz))

    J = colored_by_depth(I, lut)
    imageio.volwrite('J.tif', J, bigtiff=True)

if __name__ == '__main__':
    main()

