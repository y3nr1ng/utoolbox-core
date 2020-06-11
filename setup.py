import os

from setuptools import find_namespace_packages, setup

cwd = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(cwd, "README.md"), encoding="utf-8") as fd:
    long_description = fd.read()

setup(
    # published project name
    name="utoolbox-core",
    # from dev to release
    #   bumpversion release
    # to next version
    #   bump patch/minor/major
    version="0.0.18",
    # one-line description for the summary field
    description="Core functions for uToolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # project homepage
    url="https://github.com/liuyenting/utoolbox-core",
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
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires="~=3.7",
    # use pyproject.toml to define build system requirement
    # setup_requires=[
    # ],
    # other packages the project depends on to run
    #   install_requires -> necessity
    #   requirements.txt
    install_requires=[
        "click",
        "coloredlogs",
        "dask[complete]~=2.16.0",
        "distributed~=2.16.0",
        "h5py>=2.9",
        "humanfriendly",
        "imageio[ffmpeg]",
        "natsort",
        "numpy>=1.17",
        "pandas",
        "prompt_toolkit>=3.0",
        "tifffile",  # use the latest version, imageio bundles with older one
        "xxhash",
        "zarr",
    ],
    # additional groups of dependencies here for the "extras" syntax
    extras_require={
        # TODO remove rest of the sections
        "gpu": ["cupy-cuda101"],
        "viewer": ["napari"],
        "original": ["mako", "tqdm"],
    },
    # data files included in packages
    package_data={},
    # include all package data found implicitly
    # include_package_data=True,
    # data files outside of packages, installed into '<sys.prefix>/my_data'
    data_files=[],
    # executable scripts
    entry_points={
        "console_scripts": [
            "mm2bdv=utoolbox.cli.converter.mm2bdv:main",
            "aszarr=utoolbox.cli.aszarr:aszarr",
            "dataset=utoolbox.cli.dataset.main:dataset",
        ]
    },
    zip_safe=True,
)
