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
    version="0.0.1",
    # one-line description for the summary field
    description="A Python image processing package for LLSM.",
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
    packages=find_namespace_packages(include=["utoolbox.*"]),
    python_requires="~=3.7",
    # use pyproject.toml to define build system requirement
    # setup_requires=[
    # ],
    # other packages the project depends on to run
    #   install_requires -> necessity
    #   requirements.txt
    install_requires=[
        "dask~=2.12.0",
        "h5py>=2.9",
        "imageio[ffmpeg]",
        "numpy>=1.17",
        "pandas",
    ],
    # additional groups of dependencies here for the "extras" syntax
    extras_require={
        "gpu": ["cupy-cuda101"],
        "viewer": ["napari"],
        "original": [
            "scipy",
            # file io
            "imageio-ffmpeg",
            "tifffile",
            # parallel
            "distributed~=2.12.0",
            # utils
            "mako",
            "click",
            "coloredlogs",
            "prompt_toolkit>2.0.0",
            "tqdm",
        ],
    },
    # data files included in packages
    package_data={},
    # include all package data found implicitly
    # include_package_data=True,
    # data files outside of packages, installed into '<sys.prefix>/my_data'
    data_files=[],
    # executable scripts
    entry_points={"console_scripts": []},
    zip_safe=True,
)
