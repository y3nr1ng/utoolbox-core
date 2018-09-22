# uToolbox
[![Build Status](https://travis-ci.com/liuyenting/uToolbox.svg?token=RnNdzNQoCUCRNxtUiy7m&branch=master)](https://travis-ci.com/liuyenting/uToolbox)  
A Python image processing package for LLSM.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
It is encouraged to use environment wrapper and package manager, conda is chosen as the reference solution. Please follow the [installation section](https://conda.io/docs/user-guide/install/index.html) in their official guide.

Some of the codes require CUDA dependency, please download the binary release from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

For the time being, these are the tested version combination during development and deployment.
**TODO** add environment description

#### macOS
- High Sierra 10.13.6, Darwin 17.7.0
- CUDA 9.2.64.1

#### Windows
- Windows 7 (64-bit) SP1
- CUDA 9.2.88.1

#### Linux
- Debian 8.10 (jessie), Linux 3.12.72
- CUDA 8.0.44


### Installing
A step-by-step example demonstrate how to get a development environment running.

Clone this repository to somewhere convenient.
```
git clone https://github.com/liuyenting/utoolbox.git
cd utoolbox
```

Since pip does not honor the `setup_requires` description, we have to install basic requirements first.
```
pip install -e .
```
Then proceed with GPU related packages using `extras_require` flag.
```
pip install -e ".[gpu]"
```

**TODO** execute tests


## Deployment
Add additional notes about how to deploy this on a live system.

## Contributing

## Versioning

## Authors
- Liu, Yen-Ting

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
- [LLSpy](https://github.com/tlambert03/LLSpy) by Talley Lambert.
