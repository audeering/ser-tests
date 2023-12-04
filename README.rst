=======================================
Speech Emotion Recognition Stress Tests
=======================================

Testing suite for speech emotion recognition models.

It allows to run a large collection of tests automatically,
shows passed and failed test on the command line,
and provides detailed analysis as HTML pages.


**Tests**

All involved tests are described in the `white paper`_.


**Results**

Result overviews for all tested models
are grouped by the desired task:

* arousal_
* dominance_
* valence_
* `emotional categoris`_


**Usage**

.. note::

    At the moment,
    the tests can only be installed and run
    inside the audEERING network.
    During the next months
    we will release the required augmentation library,
    and provide ways to access the data
    and models.

If you plan to test models,
have a look at the installation_
and usage_ instructions.


.. _arousal: https://audeering.github.io/ser-tests/test/arousal.html
.. _dominance: https://audeering.github.io/ser-tests/test/dominance.html
.. _emotional categoris: https://audeering.github.io/ser-tests/test/emotion.html
.. _installation: https://audeering.github.io/ser-tests/installation.html
.. _usage: https://audeering.github.io/ser-tests/usage.html
.. _valence: https://audeering.github.io/ser-tests/test/valence.html
.. _white paper: https://audeering.github.io/ser-tests/method-tests.html
