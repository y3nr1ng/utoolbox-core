import glob
import logging
import os
from os import path
import platform
import re
import subprocess
import sys

import coloredlogs
from distutils.version import LooseVersion
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

cwd = path.abspath(path.dirname(__file__))

# get the long description from README.md
with open(path.join(cwd, "README.md"), encoding='utf-8') as fd:
    long_description = fd.read()

# CMake
class CMakeExtension(Extension):
    def __init__(self, name, src_dir=''):
        Extension.__init__(self, name, sources=[])
        self.src_dir = path.abspath(src_dir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "cmake must be installed to build the following extensions: " +
                ", ".join(ext.name for ext in self.extensions)
            )
        
        if platform.system() == 'Windows':
            cmake_ver = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1)
            )
            if cmake_ver < '3.1.0':
                raise RuntimeError('cmake >= 3.1.0 is required on Windows')
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        ext_dir = path.abspath(path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        config = 'Debug' if self.debug else 'Release'
        build_args = ['--config', config]

        if platform.system() == 'Windows':
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    config.upper(), ext_dir
                )
            ]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [
                '-DCMAKE_BUILD_TYPE=' + config
            ]
            build_args += ['--', '-j2']

        if not path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(self.build_temp)

        subprocess.check_call(
            ['cmake', ext.src_dir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, cwd=self.build_temp
        )

def convert_to_module_name(ext_dir):
    """Convert path to module annotation."""
    rel_path = path.relpath(ext_dir, cwd)
    return rel_path.replace('/', '.')

# find binary extensions by CMakeLists.txt
ext_dirs = [
    path.dirname(file_path)
    for file_path in glob.iglob(
        '{}/utoolbox/**/CMakeLists.txt'.format(cwd), recursive=True
    )
]
cmake_exts = [
    CMakeExtension(convert_to_module_name(ext_dir), ext_dir) 
    for ext_dir in ext_dirs
]

setup(
    # published project name
    name="utoolbox",

    # from dev to release
    #   bumpversion release
    # to next version
    #   bump patch/minor/major
    version='0.1.7.dev',

    # one-line description for the summary field
    description="A Python image processing package for LLSM.",

    long_description=long_description,
    long_description_content_type='text/markdown',

    # project homepage
    url="https://github.com/liuyenting/utoolbox",

    # name or organization
    author="Liu, Yen-Ting",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research'
    ],

    keywords="microscopy",

    packages=find_namespace_packages(include=['utoolbox.*']),

    python_requires='>=3.6',

    # other packages the build system would require during compilation
    setup_requires=[
    ],

    # other packages the project depends on to run
    #   install_requires -> necessity
    #   requirements.txt -> deployment (use conda environent.yml)
    install_requires=[
        # core
        'cython',
        'ipykernel',

        # numeric and processing
        'numpy',
        'scipy',
        'pandas',

        # file io
        'imageio',
        'tifffile',

        # gui
        'pyqt5',
        'plotly',

        # parallel
        'dask',
        'pycuda',

        # utils
        'mako',
        'click',
        'coloredlogs',
        'tqdm',
        'jinja2' # template engine used by pycuda
    ],

    ext_modules=cmake_exts,

    cmdclass={
        'build_ext': CMakeBuild
    },

    dependency_links=[
    ],

    # additional groups of dependencies here for the "extras" syntax
    extras_require={
    },

    # data files included in packages
    package_data={
        '': ['*.cu']
    },
    # include all package data found implicitly
    #include_package_data=True,

    # data files outside of packages, installed into '<sys.prefix>/my_data'
    data_files=[
    ],

    # executable scripts
    entry_points={
        'console_scripts': [
            'deskew=utoolbox.cli.deskew:main',
            'zpatch=utoolbox.cli.zpatch:main'
        ]
    }, 

    # cannot safely run in compressed form
    zip_safe=False
)
