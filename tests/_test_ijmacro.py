import os

import coloredlogs

from utoolbox.imagej import run_macro as _run_macro
from utoolbox.utils.decorator.macos import prelaunch_cocoa

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)


@prelaunch_cocoa
def run_macro(*args, **kwargs):
    _run_macro(*args, **kwargs)

im_path = os.path.abspath('deskew_output.tif')
run_macro('macro.ijm', im_path)
