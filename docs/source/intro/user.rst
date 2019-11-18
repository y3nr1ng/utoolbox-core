Getting Started
===============


Prerequisites
--------------
It is encouraged to use environment wrapper and package manager, conda is chosen as the reference solution. Please follow the `installation section`_ in their official guide.

Some of the codes require CUDA dependency, please download the binary release from the `NVIDIA website`_.

.. _installation section: https://conda.io/docs/user-guide/install/index.html
.. _NVIDIA website: https://developer.nvidia.com/cuda-downloads


Quick start
-----------
Create an empty workspace named `demo`, if you have your preferred environment, feel free to skip the following two lines. It is highly recommended to use `numpy` and `scipy` from `conda` instead of `pip`, since their version has `mkl`_ support directly.

.. _mkl: https://software.intel.com/en-us/mkl

.. code-block:: none

   conda create -n demo python=3 numpy scipy
   conda activate demo


Install uToolbox_ using `pip`

.. code-block:: none

   pip install utoolbox

or try out pre-release by

.. code-block:: none
  
   pip install --pre utoolbox

.. _uToolbox: https://pypi.org/project/utoolbox/

Congratz! You can now try out stuff in the guides section.

Frequently Asked Questions
==========================

What environment combinations are tested?
-----------------------------------------
+------------+------------------------------------+----------+
| Platform   | Version                            | CUDA     |
+============+====================================+==========+
|            |                                    | 9.2.88.1 |
| Windows    | Windows 7 (64-bit) SP1             +----------+
|            |                                    | 10.1.168 |
+------------+------------------------------------+----------+
| Linux      | Debian 8.10, Linux 3.12.72         | 8.0.44   |
+------------+------------------------------------+----------+
| macOS [*]_ | High Sierra 10.13.6, Darwin 17.7.0 | 9.2.64.1 |
+------------+------------------------------------+----------+

.. [*] Since I no longer owned a macOS environment, macOS support is currently stalled.