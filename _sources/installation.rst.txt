Installation
============

To install the testing framework,
you need first to clone its repository,
and change to its root path.

.. code-block:: bash

    $ git clone https://github.com/audeering/ser-tests.git
    $ cd ser-tests

Afterwards you should create a virtual environment
and install all required packages with

.. code-block:: bash

    $ pip install -r requirements.txt

For GPU support
install ``onnxruntime-gpu`` as well

.. code-block:: bash

    $ pip install onnxruntime-gpu

Before running any test,
make sure to look for a free GPU with

.. code-block::

    $ nvidia-smi

and select the free GPU (here ``0``) with
the ``--device`` argument, e.g.

.. code-block::

    $ python test.py --device "cuda:0" ...
