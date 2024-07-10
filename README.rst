=======================================
Speech Emotion Recognition Stress Tests
=======================================

Testing suite for speech emotion recognition models.

It provides a command line tool to test_
the models
for correctness,
fairness,
and robustness.
Results are grouped after the different model tasks:

* arousal_
* dominance_
* valence_
* `emotional categories`_


**🚨 Warning**:
you cannot install and run the test suite at the moment.
It depends on an internal Python package
to load all the models.
Most of the datasets used in the tests
are hosted on internal servers,
and you would need to download the datasets
from their original sources
and convert to audformat_
to run the tests.
We released the augmentation library used for the tests
as the auglib_ Python package.


.. _audformat: https://audeering.github.io/audformat/
.. _auglib: https://audeering.github.io/auglib/
.. _arousal: https://audeering.github.io/ser-tests/test/arousal.html
.. _dominance: https://audeering.github.io/ser-tests/test/dominance.html
.. _emotional categories: https://audeering.github.io/ser-tests/test/emotion.html
.. _valence: https://audeering.github.io/ser-tests/test/valence.html
.. _test: https://audeering.github.io/ser-tests/method-tests.html
