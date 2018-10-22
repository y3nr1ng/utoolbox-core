"""
Launch ImageJ as a subprocess, not shared memory in-between.
"""
import logging
import os
import subprocess as sp
import sys

__all__ = [
    'run_macro_file'
]

logger = logging.getLogger(__name__)

def run_macro_file(path, *args, headless=False, ij_root=None, plugins_dir=None):
    if not ij_root:
        logger.info("using built-in ImageJ distribution")
        cwd = os.path.dirname(__file__)
        ij_root = os.path.join(cwd, 'fiji')
        if not os.path.exists(ij_root):
            raise RuntimeError("unable to locate built-in Fiji distribution")

    if not plugins_dir:
        ij_root = os.path.dirname(ij_path)
        plugins_dir = os.path.join(ij_root, 'plugins')

    # NOTE ignore 32-bit OS
    if sys.platform.startswith('linux'):
        bin_path = 'ImageJ-linux64'
    elif sys.platform.startswith('win') or sys.platform.startswith('cygwin'):
        bin_path = 'ImageJ-win64.exe'
    elif sys.platform.startswith('darwin'):
        bin_path = os.path.join('Contents', 'MacOS', 'ImageJ-macosx')
    else:
        raise RuntimeError("unknown os")
    bin_path = os.path.join(ij_root, bin_path)

    # part 1
    bin = [
        bin_path,
        '--plugins', plugins_dir,
    ]
    # part 2
    flags = []
    if headless:
        flags += ['--headless', '--console']
    # part 3
    args = [
        '--run', path,
        ','.join([str(arg) for arg in args])
    ]
    # merge
    command = bin + flags + args
    logger.debug(' '.join(command))
    
    ret = sp.check_output(command)
