import logging
import os

import coloredlogs
import imageio

from utoolbox.imagej import Macro

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

macro = Macro()
macro.args = ['file_list', 'cal_path']
macro.batch_mode = True
macro.loop_files = True
macro.main = """
print("hello world!");
"""

print("=====\n{}=====\n".format(macro.render()))
