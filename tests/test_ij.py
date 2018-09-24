import glob
import os

import jnius_config

jnius_config.add_options('-Djava.awt.headless=true')

def discover_jars(root, recursive=True):
    return glob.glob(os.path.join(root, "*.jar"), recursive=recursive)

def set_ij_classpath(ij_root):
    jars = []
    jars.extend(discover_jars(os.path.join(ij_root, 'jars')))
    jars.extend(discover_jars(os.path.join(ij_root, 'plugins')))

    classpath = ':'.join(jars)
    print(classpath)
    jnius_config.set_classpath(classpath)

set_ij_classpath('/Applications/Fiji.app')

import jnius
ij = jnius.autoclass('net.imagej.ImageJ')

im = ij().io().Opener().open("deskew_input_2.tif")
print(im)
