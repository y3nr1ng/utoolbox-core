import os
import subprocess as sp
import sys

#TODO use subprocess solution for now (instead of pyjnius integration)
from .process import *

if 'JAVA_HOME' not in os.environ:
    if sys.platform == 'win32':
        raise OSError("please manually configure JAVA_HOME")
    else:
        java_home = sp.check_output(['/usr/libexec/java_home'])
        java_home = java_home.decode('utf-8').strip()
        if not java_home:
            raise OSError("unable to find 'java_home'")
    os.environ['JAVA_HOME'] = java_home
