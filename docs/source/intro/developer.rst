Developing uToolbox
===================
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

Prerequisites
-------------
It is encouraged to use environment wrapper and package manager, conda is chosen as the reference solution. Please follow the `installation section`_ in their official guide.

Some of the codes require CUDA dependency, please download the binary release from the `NVIDIA website`_.

For the time being, these are the tested version combination during development and deployment.

.. _installation section: https://conda.io/docs/user-guide/install/index.html
.. _NVIDIA website: https://developer.nvidia.com/cuda-downloads


Installing
----------
Following step-by-step instructions will demonstrate how to get a development environment running.

Clone this repository to somewhere convenient.

.. code-block:: none

   git clone https://github.com/liuyenting/utoolbox.git
   cd utoolbox

Install the conda environment by

.. code-block:: none

   conda env create -f environment.yml
   conda activate utoolbox-dev

this will prepare an environment with required development tools under the name `utoolbox-dev`.

Since pip does not honor the `setup_requires` description, basic requirements and native libraries are installed using conda in preivous step.

Next, we install this toolbox using editable mode

.. code-block:: none

   pip install -e .

To test the toolbox, run

.. code-block:: none

   pytest
