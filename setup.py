import os

from setuptools import find_namespace_packages, setup

cwd = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(cwd, "README.md"), encoding="utf-8") as fd:
    long_description = fd.read()

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        """
        Patch bdist_wheel to force package as platform wheel.
        
        Reference:
            https://stackoverflow.com/a/45150383
        """

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False


except ImportError:
    bdist_wheel = None

setup(
    # published project name
    name="utoolbox",
    # from dev to release
    #   bumpversion release
    # to next version
    #   bump patch/minor/major
    version="0.6.4",
    # one-line description for the summary field
    description="A Python image processing package for LLSM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # project homepage
    url="https://github.com/liuyenting/utoolbox",
    # name or organization
    author="Liu, Yen-Ting",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
    ],
    keywords="microscopy",
    packages=find_namespace_packages(include=["utoolbox.*"]),
    python_requires="~=3.7",
    # use pyproject.toml to define build system requirement
    # setup_requires=[
    # ],
    # other packages the project depends on to run
    #   install_requires -> necessity
    #   requirements.txt
    install_requires=[
        # numeric and processing
        "dask~=2.12.0",
        "numpy>=1.17",
        "scipy",
        "pandas",
        # file io
        "imageio",
        "imageio-ffmpeg",
        "tifffile",
        "h5py>=2.9",
        # parallel
        "distributed~=2.12.0",
        # utils
        "pyparsing",
        "mako",
        "click",
        "coloredlogs",
        "prompt_toolkit>2.0.0",
        "tqdm",
    ],
    # additional groups of dependencies here for the "extras" syntax
    extras_require={
        "gpu": [
            'cupy-cuda101 ; platform_system!="Darwin"',
            'cupy ; platform_system=="Darwin"',
        ],
        "viewer": ["napari"],
    },
    # data files included in packages
    package_data={"": ["*.cu"]},
    # include all package data found implicitly
    # include_package_data=True,
    # data files outside of packages, installed into '<sys.prefix>/my_data'
    data_files=[],
    # executable scripts
    entry_points={
        "console_scripts": [
            "am2csv=utoolbox.cli.am2csv:main",
            "analyze=utoolbox.cli.analyze:main",
            "dataset=utoolbox.cli.dataset:main",
            "deskew=utoolbox.cli.deskew:main",
            "mm2bdv=utoolbox.cli.mm2bdv:main",
            "zpatch=utoolbox.cli.zpatch:main",
        ]
    },
    # command hooks
    cmdclass={"bdist_wheel": bdist_wheel},
    # contains c source, cannot safely run in compressed form
    zip_safe=False,
)
