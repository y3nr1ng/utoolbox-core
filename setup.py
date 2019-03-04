import glob
import logging
import os
import platform
import re
import subprocess
import sys

from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext

cwd = os.path.abspath(os.path.dirname(__file__))

# get the long description from README.md
with open(os.path.join(cwd, "README.md"), encoding='utf-8') as fd:
    long_description = fd.read()

def wrapper_path_to_module_name(path):
    fn = os.path.basename(path)
    fn, _ = os.path.splitext(fn)
    # TODO regex wrapper
    rel_path = os.path.relpath(path, cwd)
    return rel_path.replace('/', '.')

# find all the wrappers use `wrapper_*.pyx`
wrappers = glob.glob(
    os.path.join(cwd, 'utoolbox', '**', 'wrapper_*.pyx'), recursive=True
)
# construct extensions
extensions = [
    Extension(
        wrapper_path_to_module_name(path),
        sources=[path],
        include_dirs=[os.path.join(cwd, 'utoolbox/compression/libbsc')]
    )
    for path in wrappers
]
#TODO pass attributes

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

        # utils
        'mako',
        'click',
        'coloredlogs',
        'tqdm',
        'jinja2', # template engine used by pycuda
        'xxhash'
    ],

    #ext_modules=cythonize(extensions),

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
