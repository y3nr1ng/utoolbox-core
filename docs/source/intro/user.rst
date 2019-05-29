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
Create an empty workspace named `demo`, if you have your preferred environment, feel free to skip to next part.

.. code-block:: none

   conda create -n demo python=3 numpy
   conda activate demo


Install uToolbox_ using `pip`

.. code-block:: none

   pip install utoolbox

or try out pre-release by

.. code-block:: none
  
   pip install --pre utoolbox

.. _uToolbox: https://pypi.org/project/utoolbox/


Frequently Asked Questions
==========================

What environment combinations are tested?
-----------------------------------------
Hello world!