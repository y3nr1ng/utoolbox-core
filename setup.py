from distutils.spawn import find_executable
import glob
import os
import sys

from Cython.Build import build_ext
import numpy as np
from setuptools import Extension, find_namespace_packages, setup

cwd = os.path.abspath(os.path.dirname(__file__))

###
# get description from README.md
###
with open(os.path.join(cwd, "README.md"), encoding='utf-8') as fd:
    long_description = fd.read()

###
# region FIND DEPENDENCIES
###
# numpy include directory
try:
    np_include_dir = np.get_include()
except AttributeError:
    np_include_dir = np.get_numpy_include()

# cuda 
def find_cuda_home():
    try:
        cuda_home = os.environ['CUDAHOME']
        nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    except KeyError:
        nvcc = find_executable('nvcc')
        if nvcc is None:
            raise EnvironmentError("cannot locate CUDA from PATH or CUDAHOME")
        cuda_home = os.path.dirname(os.path.dirname(nvcc))
    
find_cuda_home()

raise RuntimeError("DEBUG")
###
# endregion
###

###
# region INJECT NVCC CUSTOMIZATION
###
def customize_compiler_for_nvcc(self):
    """
    Inject deep into distutils to customize how the dispatch to gcc/nvcc works. 
    Adapt from rmcgibbo_.

    .. _rmcgibbo: https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
    """
    # tell the compiler it can process .cu
    self.src_extensions.append('.cu')

    # save references to default compiler and _compile method
    default_compiler_so, super = self.compiler_so, self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        _, ext = os.path.splitext(src)
        if ext == '.cu':
            # use cuda
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use subset of the extra_postargs
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset default compiler
        self.compiler_so = default_compiler_so
    # inject method
    self._compile = _compile

class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        super().build_extensions()
###
# endregion
###

###
# region FIND WRAPPER MODULES
###
# find all the wrappers
wrappers = glob.glob(
    os.path.join(cwd, 'utoolbox', '**', 'wrapper_*.pyx'), 
    recursive=True
)

# construct extensions
extensions = []
for abs_path in wrappers:
    # path to module name
    fn = os.path.basename(abs_path)
    fn, _ = os.path.splitext(fn)
    rel_path = os.path.relpath(abs_path, cwd)
    module = rel_path.replace(os.sep, '.')

    # external source
    root = os.path.dirname(abs_path)
    ext_include_dir = os.path.join(root, 'include')
    ext_source_dir = os.path.join(root, 'source')
    
    extension = Extension(
        module,
        sources=[],
        include_dirs=[
            np_include_dir,
            ext_include_dir
        ],
        library_dirs=[],
        runtime_library_dirs=[],
        #libraries=[], # should specify in wrapper specfial comment block 
        extra_compile_args={
            'gcc': [],
            'nvcc': [
                '-arch=sm_30',
                '--ptxas-options=-v',
                '-c',
                '--compiler-options \'-fPIC\''
            ]
        }
    )
###
# endregion
###

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

    # use pyproject.toml to define build system requirement
    #setup_requires=[
    #],

    # other packages the project depends on to run
    #   install_requires -> necessity
    #   requirements.txt
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

    cmdclass={
        'build_ext': custom_build_ext
    },

    ext_modules=extensions,

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

    # contains c source, cannot safely run in compressed form
    zip_safe=False
)
