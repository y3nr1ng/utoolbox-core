import glob
import os

import jnius_config

#jnius_config.add_options(
#    '-Djava.awt.headless=true',
#    '-Dapple.awt.UIElement=true'
#)

ij_path = 'ImageJ/ij.jar'
ij_path = os.path.abspath(ij_path)
print(ij_path)

jnius_config.set_classpath(ij_path)

import jnius

im_path = 'deskew_output_mip.tif'
im_path = os.path.abspath(im_path)
print(im_path)

String = jnius.autoclass('java.lang.String')
im_path = String(im_path)

from utoolbox.utils.decorator.macos import prelaunch_cocoa

@prelaunch_cocoa
def run_ij():
    ij = jnius.autoclass('ij.IJ')
    return ij

ij = run_ij()
ij.open()
