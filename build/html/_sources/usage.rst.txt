Usage
=====

This test suite has multiple purposes:

* :ref:`test a new model <usage-run-test>` locally
  for a particular test
  or all available tests
* :ref:`submit test results <usage-submit-results>`
  if the model is considered to be a release candidate
* inspect test results on the HTML pages
  to learn about :ref:`test details <usage-test-details>`
* :ref:`rank models <usage-rank-models>`
  by test results
  to select the next production model
* use impact analysis results and test results to summarize
  expected :ref:`differences between models <usage-model-comparison>`

We cover each of them in the following sub-sections.


.. _usage-run-test:

Run a test
----------

Tests are executed by the :file:`test.py` script,
that is located in the root folder of the repository.
It provides online help with

.. code-block:: bash

    $ python test.py --help

You always have to provide the condition to test.
You select a model with ``--model MODEL_ID``.
If no model is specified,
it will run the tests on all previously tested models.
You can also specify the maximum signal length
that should be used for the model with ``--max-signal-length``.
If you do, a short tuning parameter ID will be appended
to the model name when displaying results.
You select a test with ``--test TEST``.
If you don't specify a test
it will run all tests for the given model.
It will directly output which tests failed or passed.

**Examples**

Run :ref:`method-tests-fairness-sex` test
for the model :ref:`test-arousal-1543ec32-1.0.3`
on arousal:

.. code-block:: bash

    $ python test.py arousal --test fairness_sex --model 1543ec32-1.0.3

Run :ref:`all available tests <method-tests>`
for the model :ref:`test-arousal-1543ec32-1.0.3`
on arousal:

.. code-block:: bash

    $ python test.py arousal --model 1543ec32-1.0.3

Run :ref:`all available tests <method-tests>`
for the model with ID 1543ec32-1.0.3
on arousal with a maximum signal length of 3 seconds:

.. code-block:: bash

    $ python test.py arousal --model 1543ec32-1.0.3 --max-signal-length 5

Run :ref:`method-tests-fairness-sex` test
for :ref:`all previously tested arousal models <test-arousal>`

.. code-block:: bash

    $ python test.py arousal --test fairness_sex

Run :ref:`all available tests <method-tests>`
for :ref:`all previously tested arousal models <test-arousal>`

.. code-block:: bash

    $ python test.py arousal


.. _usage-submit-results:

Submit results
--------------

The results of a test are stored
under :file:`docs/results/test/{CONDITION}/{MODEL_ID}/{TEST}`
as CSV and PNG files.
For example,
the folder
:file:`docs/results/test/arousal/1543ec32-1.0.3/correctness_regression/`
contains besides other files
:file:`mean-squared-error.csv`.

If this is the first time you tested the selected model
it will also store information about that model
under the model folder
:file:`docs/results/test/{CONDITION}/{MODEL_ID}`.

If your tested model is ranked under the top five
condition overview pages,
or you think there are other reasons worth submitting
the test results,
please commit those files to a new branch,
push to the Github server,
and open a pull request.


.. _usage-test-details:

Test details as HTML
--------------------

Every time you push to the ``main`` branch
of the Github repository,
a CI job will automatically update the HTML pages
you find under
https://audeering.github.io/ser-tests/.
Here,
you can inspect all submitted test results.

To build that page locally, please run:

.. code-block:: bash

    $ pip install -r docs/requirements.txt
    $ python -m sphinx docs/ build/html -b html


.. _usage-rank-models:

Rank models
-----------

Models are automatically ranked
by their test results
on the test overview pages.
The ranking is calculated
by the percentage of passed tests.


.. _usage-model-comparison:

Model comparisons
-----------------

If we want to update compare one model
to another,
we need to summarize the changes
the user might expect.
To this end, we show the comparison of the individual
test results.

If a change from one model baseline to one
or more model candidates should be analysed,
the tests of the involved models have to be
:ref:`run <usage-run-test>` and
:ref:`submitted <usage-submit-results>`.

Finally, the intended baseline and candidate
model ids have to be specified in
:file:`docs/results/comparison/{CONDITION}.yaml` under the
respective condition in order for them
to be displayed in the HTML pages.
For example, to show the comparison between two models
1543ec32-1.0.3 and 51c582b7-1.0.0
for emotion
the file :file:`comparison/emotion.yaml` should contain:

.. code-block:: yaml

    - baseline: 1543ec32-1.0.3
      candidates:
        - 51c582b7-1.0.0
