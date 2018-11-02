
import os

import imageio
from skimage.transform import rescale
from tqdm import tqdm

from utoolbox.container import ImageDatastore

src_dir = 'G:/live_localization/09142018_HalotagEGFPc3Nup153+PmeI/clone1'
def read_func(x):
    return (x, imageio.volread(x))
imds = ImageDatastore(src_dir, read_func)

dst_dir = 'G:/live_localization/09142018_HalotagEGFPc3Nup153+PmeI/clone1_downsample2'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

for path, im in tqdm(imds):
    im_out = rescale(im, .5, multichannel=False, preserve_range=True)
    name = os.path.basename(path)
    path = os.path.join(dst_dir, name)
    imageio.volwrite(path, im_out.astype(im.dtype))
