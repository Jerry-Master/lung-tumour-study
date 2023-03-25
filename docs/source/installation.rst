Installation
============

First of all, create a new python environment and activate it. This library has been tested for python 3.8, 3.9, 3.10.

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate

Now, install the tumourkit library.

.. code-block:: console
    
   $ pip install tumourkit

Before you can start using the library, there are some external libraries not uploaded to pypi that need to be taken into account. The first is imgaug:

.. code-block:: console
    
   $ pip install git+https://github.com/marcown/imgaug.git@74e63f2#egg=imgaug

The other two libraries that you need to install are `Pytorch <https://pytorch.org/>`_ and `Deep Graph Library <https://www.dgl.ai/>`_. Depending on which operating system you use, and whether you have a GPU or not, the installation changes.
To install them, follow the instructions on their official page.

Known errors
------------

Mixed CPU - GPU installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you try to use the GPU version of Deep Graph Library but have installed the CPU you will receive the following error.

.. code-block:: console
    
   Check failed: allow_missing: Device API gpu is not enabled. Please install the cuda version of dgl.


Just install the GPU build and you will be fine.

Missing CUDA libraries
^^^^^^^^^^^^^^^^^^^^^^

If you come accross something like

.. code-block:: console
    
   OSError: libcusparse.so.11: cannot open shared object file: No such file or directory


That means your python environment does not link correctly to your CUDA installation. You will have to edit the :code:`LD_LIBRARY_PATH` environmental variable so that the dynamic library :code:`libcusparse.so.11` can be found. Typically it is found under :code:`nvidia/cublas/lib` so a possible fix is

.. code-block:: console
    
   $ export LD_LIBRARY_PATH=.venv/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
