.. plot::
    :context: reset
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    import audeer
    import audiofile
    import audplot

    blue = '#3277b4'
    orange = '#f27e13'
    red = '#e13b41'

    scale = .85
    plt.rcParams['figure.figsize'] = (scale * 6.4, scale * 4.8)
    plt.rcParams['font.size'] = 13

    media_dir = audeer.path('extra', 'media')

    def plot(signal, color, text):
        signal = np.atleast_2d(signal)
        signal = signal[0, :]
        fig, ax = plt.subplots(1, figsize=(4, 1.4))
        audplot.waveform(
            signal,
            text=text,
            color=color,
            ax=ax,
        )
        plt.tight_layout()


.. _method-tests:

Method Tests
============

Here we discuss
all the involved tests,
their metrics, thresholds,
and the motivation behind the tests.


.. _method-tests-introduction:

Introduction
------------

We follow :cite:t:`Zhang2019`
and group our tests
under the three categories
**correctness**,
**fairness**,
**robustness**.
The correctness tests
try to make sure
that the predictions of the model
on the input signal
follow the truth
as closely as possible
for different metrics.
The fairness test
looks into sub-groups of the data
like sex or accent
and ensures
that the model behaves similar
for all sub-groups.
The robustness tests investigate
how much the model output is affected
by changes to input signal
like low pass filter
or changes in gain.


.. =======================================================================
.. _method-tests-random-model:

Random Model
------------

To provide some reference values for the test results
we add a random model. For categorical emotion, the
**random-categorical** model randomly samples from
a uniform categorical distribution to generate the prediction.
For dimensional emotion, the **random-gaussian** model
randomly samples from a truncated Gaussian distribution
with values between :math:`0` and :math:`1`,
a mean value of :math:`0.5`, and a standard deviation of :math:`\frac{1}{6}`.


.. =======================================================================
.. _method-tests-fairness-thresholds:

Fairness Thresholds
-------------------

For all fairness tests,
we use simulations based on random models as reference
for setting test thresholds,
since a random model has no bias towards certain groups.
We simulate the tests by running a random model
on random predictions 1000 times under different
conditions (number of samples per fairness group,
number of fairness groups). The maximum value that
occurs in the set of metric results is then used as reference
for the respective threshold.

For regression tasks we use the distribution of the
**random-gaussian** model to simulate the predictions as well as the
ground truth.
For categories, we use the distribution of the (uniform)
**random-categorical** model to simulate the predictions.
For the simulation of categorical ground truth, we simulate both a
uniform distribution as well as a sparse distribution with
the class probabilities :math:`(0.05, 0.05, 0.3, 0.6)`, and select the
distribution that applies to the respective test.

For certain test sets, the distribution of the ground truth for
certain groups varies from the distribution of other groups.
The maximum difference in prediction in the simulation increases
under such biases. Therefore we balance the test sets with ground
truth labels by selecting 1000 samples from the group with the
fewest samples, and 1000 samples from each other group with
similar truth samples.
For certain regression test sets this results
in certain regression bins having very few samples,
no longer matching the assumed Gaussian distribution.
In these cases, for fairness metrics that involve bins,
we skip bins with too few samples. 
We set the minimum number of samples :math:`n_{\text{bin}}` to the expected
number of samples in the first bin for a Gaussian distribution
with a mean of :math:`0.5` and a standard deviation of :math:`\frac{1}{6}`:

:math:`n_{\text{bin}} = \mathbb{P}(X\leq0.25) \cdot n`,

where :math:`n` is the total number of samples,
and the random variable :math:`X` follows the aforementioned distribution.

We take the same approach for the tests with unlabelled test sets
in the case that a model has very few predictions in a certain bin
for the combined test set.


.. =======================================================================
.. _method-tests-correctness-classification:

Correctness Classification
--------------------------

The correctness classification tests
include standard metrics
that are used to evaluate classification problems,
namely
**Precision Per Class**,
**Recall Per Class**,
**Unweighted Average Precision**,
**Unweighted Average Recall**.

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/correctness_classification.csv


.. =======================================================================
.. _method-tests-correctness-consistency:

Correctness Consistency
-----------------------

The correctness consistency tests
check whether the models' predictions
on other tasks are consistent with
the expected result.
For example, we know from the
literature that happiness is characterized
by high valence and that fear tends to coincide with low
dominance :cite:p:`Fontaine2007`.
Based on comparing various literature results
:cite:p:`Fontaine2007,Hoffmann2012,Gillioz2016,Verma2017`,
we expect the following dimensional values
for emotional categories:

.. csv-table:: Emotion categories and their dimensional ranges
    :header-rows: 1
    :file: method-tests/correctness_consistency_ranges.csv

The **Samples in Expected High Range** test
checks whether the proportion
of samples which are expected
to have a high value and have a prediction
>=0.55 is above a given threshold.

The **Samples in Expected Low Range** test
checks whether the proportion
of samples which are expected
to have a low value and have a prediction
<= 0.45 is above a given threshold.

The **Samples in Expected Neutral Range** test
checks whether the proportion
of samples which are expected
to have a neutral value and have a prediction
in the range of 0.3 and 0.6 is above a given threshold.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/correctness_consistency.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/correctness_consistency.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/correctness_consistency.csv


.. =======================================================================
.. _method-tests-correctness-distribution:

Correctness Distribution
------------------------

The distributions
as returned from the model
for the different test sets
should be very similar
to the gold standard distributions.

The **Jensen Shannon Distance**
(compare `Jensen-Shannon divergence`_)
provides a single value
to judge the distance between
two random distributions.
The value ranges from 0 to 1,
with lower values indicating
a more similar distribution.
We bin the distributions
into 10 bins
before calculating the distance.

The test **Relative Difference Per Class**
checks that the number of samples
per class is comparable
between the model prediction
and the gold standard.
We measure the difference of the number of samples
in relative terms
compared to the overall number of samples
in the test set.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/correctness_distribution.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/correctness_distribution.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/correctness_distribution.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/correctness_distribution.csv


.. =======================================================================
.. _method-tests-correctness-regression-per-segment:

Correctness Regression
----------------------

The correctness regression tests
include standard metrics
that are used to evaluate regression problems,
namely
**Concordance Correlation Coeff**,
**Pearson Correlation Coeff**,
**Mean Absolute Error**.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/correctness_regression.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/correctness_regression.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/correctness_regression.csv


.. =======================================================================
.. _method-tests-correctness-speaker-average:

Correctness Speaker Average
---------------------------

The models should be able to
estimate the correct average value per speaker.
For the classification task, the class proportions
should be estimated correctly for each speaker.

We only consider speakers with at least 10 samples
for regression, and with at least 8 samples per class
for classification.

The test
**Mean Absolute Error**
measures the absolute error per speaker.

For the classification task,
the test
**Class Proportion Mean Absolute Error**
measures the absolute error in the
predicted proportion of each class
per speaker.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/correctness_speaker_average.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/correctness_speaker_average.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/correctness_speaker_average.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/correctness_speaker_average.csv


.. =======================================================================
.. _method-tests-correctness-speaker-ranking:

Correctness Speaker Ranking
---------------------------

For some applications,
it may be of interest to create a ranking
of speakers in order to spot outliers on
either side of the ranking.

The test uses the raw values per sample
to calculate the average value for each speaker for regression.
For classification, the test uses the proportions
per class for each speaker.

We only consider speakers with at least 10 samples
for regression, and with at least 8 samples per class
for classification.

For the :ref:`method-tests-correctness-speaker-ranking` part,
keep in mind
that these are relative scores
and do not represent the absolute accuracy of the prediction.
That is why we have the :ref:`method-tests-correctness-speaker-average` part.

As a measure
of the overall ranking
we use
`Spearman's rank correlation coefficient`_
(**Spearmans Rho**),
which ranges from 0 to 1.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/correctness_speaker_ranking.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/correctness_speaker_ranking.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/correctness_speaker_ranking.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/correctness_speaker_ranking.csv


.. =======================================================================
.. _method-tests-fairness-accent:

Fairness Accent
---------------

The models should not show a bias
regarding the accent of a speaker.
For now, we only investigate English accents.

The investigation is based
on the `speech-accent-archive`_ database,
which provides recordings
for several different accents.
Each speaker in the database was asked
to read the same English paragraph,
lasting a little longer than 3 minutes
in most cases.
The database also includes speakers
with English as their native language.
For each of the 31 accents,
there are at least 60 audio samples.

To test the different accents
predictions for recordings
from speakers with different native languages
were collected
and compared to the combined database.
The accent was named after their native language.
For each accent
we use recordings from 5 female
and 5 male speakers.

The **Mean Value** over all samples
should not change for any specific accent
compared to the combined database.

For the test **Relative Difference Per Bin**
we follow :cite:t:`Agarwal2019` and
discretize the regression model outputs
into 4 bins. 
The test
checks that the number of samples
per bin is comparable
between the model prediction for
one accent
and the model prediction for
the combined database.
We measure the difference of the number of samples
in relative terms
compared to the overall number of samples
in the test set.

The test **Relative Difference Per Class**
checks that the number of samples
per class is comparable
between the model prediction for
one accent
and the model prediction for
the combined database.
We measure the difference of the number of samples
in relative terms
compared to the overall number of samples
in the test set.

We base the thresholds on simulations
with a random-categorical and a
random-gaussian model for
30 fairness groups and 60 samples
per group.
For the test Relative Difference Per Bin
we require at least 4 predictions per bin
in the combined dataset,
or we skip that bin.

.. figure:: extra/media/fairness_thresholds/plots/max_mean.png

    The maximum difference in mean value
    for a random gaussian model from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_relative_difference_per_bin.png

    The maximum difference in relative difference per bin
    for a random gaussian model from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_uniform_relative_difference_per_class.png

    The maximum difference in relative difference per class
    for a random uniform categorical model from 1000
    simulations.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/fairness_accent.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/fairness_accent.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/fairness_accent.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/fairness_accent.csv


.. _speech-accent-archive: https://www.kaggle.com/rtatman/speech-accent-archive


.. =======================================================================
.. _method-tests-fairness-language:

Fairness Language
-----------------

The models should not show a bias
regarding the language of a speaker.
As the perceived emotion is not independent
of language and culture
we don't expect it
to be without bias
for all languages.
In this test
we focus on the main languages
for which the model should be applied.

For each of the languages
English,
German,
Italian,
French,
Spanish,
and Chinese,
2000 random samples
are selected.
The prediction of the combined data
is then compared
against the prediction
for each individual language.

The **Mean Value** over all samples
should not change for any specific language
compared to the combined database.

For the test **Relative Difference Per Bin**
we follow :cite:t:`Agarwal2019` and
discretize the regression model outputs
into 4 bins. 
The test
checks that the number of samples
per bin is comparable
between the model prediction for
one language
and the model prediction for
the combined database.
We measure the difference of the number of samples
in relative terms
compared to the overall number of samples
in the test set.

The test **Relative Difference Per Class**
checks that the number of samples
per class is comparable
between the model prediction for
one language
and the model prediction for
the combined database.
We measure the difference of the number of samples
in relative terms
compared to the overall number of samples
in the test set.

We base the thresholds on simulations
with a random-categorical and a
random-gaussian model for
6 fairness groups and at least 1000 samples
per group, and increase them to accomodate
for potential variations of the ground truth for different languages
in the database.
For the test Relative Difference Per Bin
we require at least 67 predictions per bin
in the combined dataset,
or we skip that bin.

.. figure:: extra/media/fairness_thresholds/plots/max_mean.png

    The maximum difference in mean difference
    for a random gaussian model from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_relative_difference_per_bin.png

    The maximum difference in relative difference per bin
    for a random gaussian model from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_uniform_relative_difference_per_class.png

    The maximum difference in relative difference per class
    for a random uniform categorical model from 1000
    simulations.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/fairness_language.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/fairness_language.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/fairness_language.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/fairness_language.csv


.. =======================================================================
.. _method-tests-fairness-linguistic-sentiment:

Fairness Linguistic Sentiment
-----------------------------

The models should not show a bias
regarding the language of a speaker.
This also extends to the text sentiment
that is contained in a sample.
If the text content has an influence
on the model predictions,
it should have the same influence for each
language.

We use the checklist-synth database,
which contains synthetic speech
of text with sentiment-labelled
sentences or words generated from `checklist`_.
The text was generated from the
English sentiment testing suite, and
then translated into multiple languages.
For each language, a publicly available
speech-to-text model using both the
libraries `TTS`_ and `espnet`_
was used to synthesize the audio samples corresponding
to the text.

For each of the languages
German,
English,
Spanish,
French,
Italian,
Japanese,
Portuguese,
and Chinese
up to 2000 random samples
are selected per test set.
The prediction of the combined data
is then compared
against the prediction
for each individual language.

For this test we only want to measure
the influence of text sentiment
for different languages, and not
general language biases, which are
covered in the
:ref:`method-tests-fairness-language`
test.
Therefore, we compare the shift in
prediction when filtering the samples
by a specific sentiment. We
denote all samples with sentiment :math:`s` and
language :math:`l` as :math:`X_{l, s}`,
and all combined samples of language :math:`l`
as :math:`X_l`.
We compute the
difference between the
shift in prediction
for a certain sentiment
and language
and the average of the shifts
in prediction for that sentiment
for all languages :math:`l_i, 1 \leq i \leq L`

.. math::

    \text{shift}(X_{l, s})
    - \frac{1}{L}\sum_{i=1}^{L} \text{shift}(X_{l_i, s}).

The **Mean Shift Difference Positive Sentiment**,
**Mean Shift Difference Negative Sentiment**, and
**Mean Shift Difference Neutral Sentiment** tests
compute the difference between
the shift in mean value for one language
and the average shift in mean value across all languages.
They ensure that its absolute value
is below a given threshold.
The shift function of the tests is given by

.. math::

    \text{shift}_{\text{mean}}(X_{l, s}) =
    \text{mean}(\text{prediction}(X_{l, s})) -
    \text{mean}(\text{prediction}(X_l)).

For the tests **Bin Proportion Shift Difference Positive Sentiment**,
**Bin Proportion Shift Difference Negative Sentiment**, and
**Bin Proportion Shift Difference Neutral Sentiment**
we follow :cite:t:`Agarwal2019` and
discretize the regression model outputs
into 4 bins.
The tests
compute the difference between
the shift in bin proportion for one language
and the average shift in bin proportion across all languages.
They ensure that its absolute value
is below a given threshold.
The shift function of the tests is given by

.. math::
    \begin{align*}
        \text{shift}_{b}(X_{l,s}) =
        & \frac{1}{| X_{l,s} |} |
            \{ y \; | \; y = b \; \text{and} \; y \in \text{prediction}_{\text{bin}}(X_{l, s}) \}
        | - \\
        & \frac{1}{ | X_l |} |
            \{ y \; | \; y = b \; \text{and} \; y \in \text{prediction}_{\text{bin}}(X_{l}) \}
        |,
    \end{align*}

where :math:`b` is the tested bin
and :math:`\text{prediction}_{\text{bin}}` is a function
that applies the model to a given set of samples
and assigns a bin label to each of the model outputs.

The tests **Class Proportion Shift Difference Positive Sentiment**,
**Class Proportion Shift Difference Negative Sentiment**, and
**Class Proportion Shift Difference Neutral Sentiment**
compute the difference between
the shift in class proportion for one language
and the average shift in class proportion across all languages.
They ensure that its absolute value
is below a given threshold.
The shift function of the tests is given by

.. math::
    \begin{align*}
        \text{shift}_{c}(X_{l,s}) =
        & \frac{1}{| X_{l,s} |} |
            \{ y \; | \; y = c \; \text{and} \; y \in \text{prediction}(X_{l, s}) \}
        | - \\
        & \frac{1}{ | X_l |} |
            \{ y \; | \; y = c \; \text{and} \; y \in \text{prediction}(X_{l}) \}
        |,
    \end{align*}

where :math:`c` is the tested class label.

We base the thresholds on simulations
with a random-categorical and a
random-gaussian model for
8 fairness groups and at least 1000 samples
per group.
For the Bin Proportion Shift Difference tests
we require at least 67 predictions per bin
in the combined dataset per sentiment,
or we skip the bin for that sentiment.

.. figure:: extra/media/fairness_thresholds/plots/max_mean_shift_diff.png

    The maximum difference in mean shift difference
    for a random gaussian model from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_bin_shift_diff.png

    The maximum difference in bin proportion shift
    for a random gaussian model from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_uniform_class_shift_diff.png

    The maximum difference in class proportion shift
    for a random uniform categorical model from 1000
    simulations.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/fairness_linguistic_sentiment.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/fairness_linguistic_sentiment.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/fairness_linguistic_sentiment.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/fairness_linguistic_sentiment.csv


.. _checklist: https://github.com/marcotcr/checklist
.. _espnet: https://github.com/espnet/espnet
.. _TTS: https://github.com/coqui-ai/TTS


.. =======================================================================
.. _method-tests-fairness-pitch:

Fairness Pitch
--------------

The models should not show a bias
regarding the average pitch of a speaker.

We only include speakers with more than 25 samples
for this test.
For each of these speakers, we compute the pitch
of each sample.
For pitch estimation we extract F0 framewise with praat_
and calculate a mean value for each segment,
ignoring frames with a pitch value of 0 Hz.
We exclude segments from the analysis
that show a F0 below 50 Hz
or above 350 Hz
to avoid pitch estimation outlier
to influence the tests.
We then compute the average of all samples
belonging to a speaker, and assign one of
3 pitch groups to that speaker.
The low pitch group is assigned to speakers
with an average pitch less than or equal to 145 Hz,
the medium pitch group to speakers with an average
pitch of more than 145 Hz but less than or equal to 190 Hz, and
the high pitch group to speakers with an average
pitch higher than 190 Hz.

We use two kinds of fairness criteria for this test.
Firstly, we ensure that the performance
for each pitch group is similar
to the performance for the entire test set.
Secondly, based on the principle of 
*Equalized Odds* :cite:p:`Mehrabi2021` we
ensure that we have similar values
between each group and the entire test set
for recall and precision for certain output classes.

The test thresholds are affected
if the ground truth labels show a bias
for a particular pitch group.
To avoid this
we first balance the test sets
by selecting 1000 samples randomly
from the pitch group with the fewest samples,
and 1000 samples from the other pitch groups
with similar truth values.

The **Concordance Correlation Coeff High Pitch**,
**Concordance Correlation Coeff Low Pitch**, and
**Concordance Correlation Coeff Medium Pitch** tests
ensure that the difference
in concordance correlation coefficient
between the respective pitch group and
the combined test set
is below the given threshold.

For the tests **Precision Per Bin High Pitch**,
**Precision Per Bin Low Pitch**, and
**Precision Per Bin Medium Pitch**
we
discretize the regression model outputs
into 4 bins and require
that the difference in
precision per bin
between the respective pitch group and
the combined test set
is below the given threshold.

The **Precision Per Class High Pitch**,
**Precision Per Class Low Pitch**, and
**Precision Per Class Medium Pitch** tests
ensure that the difference
in precision per class
between the respective pitch group and
the combined test set
is below the given threshold.

For the tests **Recall Per Bin High Pitch**,
**Recall Per Bin Low Pitch**, and
**Recall Per Bin Medium Pitch**
we
discretize the regression model outputs
into 4 bins and require
that the difference in
recall per bin
between the respective pitch group and
the combined test set
is below the given threshold.

The **Recall Per Class High Pitch**,
**Recall Per Class Low Pitch**, and
**Recall Per Class Medium Pitch** tests
ensure that the difference
in recall per class
between the respective pitch group and
the combined test set
is below the given threshold.

The **Unweighted Average Recall High Pitch**,
**Unweighted Average Recall Low Pitch**, and
**Unweighted Average Recall Medium Pitch** tests
ensure that the difference
in unweighted average recall
between the respective pitch group and
the combined test set
is below the given threshold.

We base the thresholds
on simulations
with a random-categorical and a
random-gaussian model for
3 fairness groups and 1000 samples
per group, and assume a sparse 
distribution of the ground truth for categories.
For the Precision Per Bin and Recall Per Bin tests
we require at least 67 samples per bin
in the ground truth of the combined dataset,
or we skip that bin.

.. figure:: extra/media/fairness_thresholds/plots/max_ccc.png

    The maximum difference in CCC
    for a random gaussian model on a
    random gaussian ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_precision_per_bin.png

    The maximum difference in precision per bin
    for a random gaussian model on a
    random gaussian ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_truthsparse_preduniform_precision_per_class.png

    The maximum difference in precision per class
    for a random uniform categorical model on a
    random sparse categorical ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_recall_per_bin.png

    The maximum difference in recall per bin
    for a random gaussian model on a
    random gaussian ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_truthsparse_preduniform_recall_per_class.png

    The maximum difference in precision per class
    for a random uniform categorical model on a
    random sparse categorical ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_truthsparse_preduniform_recall.png

    The maximum difference in UAR
    for a random uniform categorical model on a
    random sparse categorical ground truth from 1000
    simulations.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/fairness_pitch.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/fairness_pitch.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/fairness_pitch.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/fairness_pitch.csv


.. =======================================================================
.. _method-tests-fairness-sex:

Fairness Sex
------------

The models should not show a bias
regarding the sex of a speaker.

We use two kinds of fairness criteria for this test.
Firstly, we ensure that the performance
for each sex is similar
to the performance for the entire test set.
Secondly, based on the principle of 
*Equalized Odds* :cite:p:`Mehrabi2021` we
ensure that we have similar values
between each sex and the entire test set
for recall and precision for certain output classes.

The test thresholds are affected
if the ground truth labels show a bias
for a particular sex.
To avoid this
we first balance the test sets
by selecting 1000 samples randomly
from the sex group with the fewest samples,
and 1000 samples from the other sex group
with similar truth values.

The **Concordance Correlation Coeff Female** and
**Concordance Correlation Coeff Male** tests
ensure that the difference
in concordance correlation coefficient
between the respective sex and
the combined test set
is below the given threshold.

For the tests **Precision Per Bin Female** and
**Precision Per Bin Male**
we
discretize the regression model outputs
into 4 bins and require
that the difference in
precision per bin
between the respective sex and
the combined test set
is below the given threshold.

The **Precision Per Class Female** and
**Precision Per Class Male** tests
ensure that the difference
in precision per class
between the respective sex and
the combined test set
is below the given threshold.

For the tests **Recall Per Bin Female** and
**Recall Per Bin Male**
we
discretize the regression model outputs
into 4 bins and require
that the difference in
recall per bin
between the respective sex and
the combined test set
is below the given threshold.

The **Recall Per Class Female** and
**Recall Per Class Male** tests
ensure that the difference
in recall per class
between the respective sex and
the combined test set
is below the given threshold.

The **Unweighted Average Recall Female** and
**Unweighted Average Recall Male** tests
ensure that the difference
in unweighted average recall
between the respective sex and
the combined test set
is below the given threshold.

We base the thresholds
on simulations
with a random-categorical and a
random-gaussian model for
2 fairness groups and 1000 samples
per group, and assume a sparse 
distribution of the ground truth for categories.
For the Precision Per Bin and Recall Per Bin tests
we require at least 67 samples per bin
in the ground truth of the combined dataset,
or we skip that bin.

.. figure:: extra/media/fairness_thresholds/plots/max_ccc.png

    The maximum difference in CCC
    for a random gaussian model on a
    random gaussian ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_precision_per_bin.png

    The maximum difference in precision per bin
    for a random gaussian model on a
    random gaussian ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_truthsparse_preduniform_precision_per_class.png

    The maximum difference in precision per class
    for a random uniform categorical model on a
    random sparse categorical ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_recall_per_bin.png

    The maximum difference in recall per bin
    for a random gaussian model on a
    random gaussian ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_truthsparse_preduniform_recall_per_class.png

    The maximum difference in recall per class
    for a random uniform categorical model on a
    random sparse categorical ground truth from 1000
    simulations.

.. figure:: extra/media/fairness_thresholds/plots/max_truthsparse_preduniform_recall.png

    The maximum difference in UAR
    for a random uniform categorical model on a
    random sparse categorical ground truth from 1000
    simulations.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/fairness_sex.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/fairness_sex.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/fairness_sex.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/fairness_sex.csv


.. =======================================================================
.. _method-tests-robustness-background-noise:

Robustness Background Noise
---------------------------

We show in the :ref:`method-tests-robustness-small-changes` test,
that our emotion models' predictions might change
when adding white noise to the input signal.
Similar results are known from the speech emotion recognition literature.
:cite:t:`Jaiswal2021` have shown
that adding environmental noise like rain or coughing,
leads to a drop of performance of around 50%
for a signal-to-noise ratio of 20 dB.

The purpose of this test is to investigate
how the model performance is influenced
by different added noises
at lower signal-to-noise ratios
as used in the :ref:`method-tests-robustness-small-changes` test.
As background noises we use the following:

* Babble Noise: 4 to 7 speech samples from the speech table
  of the musan_ database :cite:`Snyder2015`
  are mixed
  and added with an SNR of 20 dB
* Coughing: one single cough
  from our internal cough-speech-sneeze_ database
  (based on :cite:`Amiriparian2017`)
  is added to each sample
  at a random position
  with an SNR of 10 dB
* Environmental Noise: a noise sample from the noise table
  of the musan_ database
  is added with an SNR of 20 dB.
  The noise table includes technical noises,
  such as DTMF tones,
  dialtones,
  fax machine noises,
  and more,
  as well as ambient sounds,
  such as car idling,
  thunder,
  wind,
  footsteps,
  paper rustling,
  rain,
  animal noises
* Music: a music sample from the music table
  of the musan_ database
  is added with an SNR of 20 dB.
  The music table includes Western art music
  (e.g. Baroque, Romantic, Classical)
  and popular genres (e.g. jazz, bluegrass, hiphop)
* Sneezing: one single sneeze
  from the cough-speech-sneeze_ database
  is added to each sample
  at a random position
  with an SNR of 10 dB
* White Noise: white noise
  is added with an SNR of 20 dB

.. Plot impulse responses and include listening examples

.. Speech
.. plot::
    :context: close-figs

    from common.robustness_background_noise import noise_transform

    out_dir = audeer.path(media_dir, 'robustness_background_noise')
    audeer.mkdir(out_dir)

    speech_file = audeer.path(media_dir, 'speech.wav')
    speech, sampling_rate = audiofile.read(speech_file, always_2d=True)
    plot(speech, red, 'Original\nAudio')

.. raw:: html

    <p><audio controls src="media/speech.wav"></audio></p>

.. Babble Noise
.. plot::
    :context: close-figs

    babble_noise = noise_transform(
        speech,
        sampling_rate,
        'babble',
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'babble_noise.wav'),
        babble_noise,
        sampling_rate,
    )
    plot(babble_noise, blue, 'Babble\nNoise')

.. raw:: html

    <p><audio controls src="media/robustness_background_noise/babble_noise.wav"></audio></p>

.. Coughing
.. plot::
    :context: close-figs

    coughing = noise_transform(
        speech,
        sampling_rate,
        'coughing',
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'coughing.wav'),
        coughing,
        sampling_rate,
    )
    plot(coughing, blue, 'Coughing')

.. raw:: html

    <p><audio controls src="media/robustness_background_noise/coughing.wav"></audio></p>

.. Environmental Noise
.. plot::
    :context: close-figs

    environmental_noise = noise_transform(
        speech,
        sampling_rate,
        'environmental',
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'environmental_noise.wav'),
        environmental_noise,
        sampling_rate,
    )
    plot(environmental_noise, blue, 'Environmental\nNoise')

.. raw:: html

    <p><audio controls src="media/robustness_background_noise/environmental_noise.wav"></audio></p>

.. Music
.. plot::
    :context: close-figs

    music = noise_transform(
        speech,
        sampling_rate,
        'music',
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'music.wav'),
        music,
        sampling_rate,
    )
    plot(music, blue, 'Music')

.. raw:: html

    <p><audio controls src="media/robustness_background_noise/music.wav"></audio></p>

.. Sneezing
.. plot::
    :context: close-figs

    sneezing = noise_transform(
        speech,
        sampling_rate,
        'sneezing',
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'sneezing.wav'),
        sneezing,
        sampling_rate,
    )
    plot(sneezing, blue, 'Sneezing')

.. raw:: html

    <p><audio controls src="media/robustness_background_noise/sneezing.wav"></audio></p>

.. White Noise
.. plot::
    :context: close-figs

    white_noise = noise_transform(speech, sampling_rate, 'white')
    audiofile.write(
        audeer.path(out_dir, 'white_noise.wav'),
        white_noise,
        sampling_rate,
    )
    plot(white_noise, blue, 'White\nNoise')

.. raw:: html

    <p><audio controls src="media/robustness_background_noise/white_noise.wav"></audio></p>


.. Extra space

|

The **Change CCC Babble Noise**,
**Change CCC Coughing**,
**Change CCC Environmental Noise**,
**Change CCC Music**,
**Change CCC Sneezing**,
and **Change CCC White Noise**
tests ensure
that the Concordance Correlation Coefficient (CCC)
does not decrease
too much when adding
the given background noise.

The **Change UAR Babble Noise**,
**Change UAR Coughing**,
**Change UAR Environmental Noise**,
**Change UAR Music**,
**Change UAR Sneezing**,
and **Change UAR White Noise**
tests ensure
that the Unweighted Average Recall (UAR)
does not decrease
too much when adding
the given background noise.

The **Percentage Unchanged Predictions Babble Noise**,
**Percentage Unchanged Predictions Coughing**,
**Percentage Unchanged Predictions Environmental Noise**,
**Percentage Unchanged Predictions Music**,
**Percentage Unchanged Predictions Sneezing**,
**Percentage Unchanged Predictions White Noise**
tests check
that the percentage of samples with
unchanged predictions is high enough
when adding the given background noise.
We use the same definitions as in the
:ref:`method-tests-robustness-small-changes`
to compute this percentage.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/robustness_background_noise.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/robustness_background_noise.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/robustness_background_noise.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/robustness_background_noise.csv


.. _musan: http://www.openslr.org/17/
.. _cough-speech-sneeze: https://audeering.github.io/datasets/datasets/cough-speech-sneeze.html


.. _method-test-robustness-low-quality-phone:

Robustness Low Quality Phone
----------------------------

The models should be robust
to a low quality phone recording condition.
Low quality phone recordings usually
have stronger compression,
and coding artifacts.
In addition,
they may show low pass behavior
as indicated by the following plot
showing the magnitude spectrum
for one low quality phone sample from switchboard-1_,
and a high quality headphone recording sample from emovo.

.. Prepare helper functions
.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    import audiofile

    def load_audio(table):
        signals = []
        for file, start, end in zip(
                table.files,
                table.starts,
                table.ends,
        ):
            if not pd.isna(end):
                duration = (end-start).total_seconds()
            else:
                duration = None
            signal, _ = audiofile.read(
                file,
                offset=start.total_seconds(),
                duration=duration,
            )
            signals.append(signal)
        return np.concatenate(signals)
        

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')


    def plot_spectrum(high_qual, low_qual, sampling_rate, low_qual_offset=None):
        # Adjust signal levels
        rms_high_qual = np.sqrt(np.mean(high_qual ** 2))
        rms_low_qual = np.sqrt(np.mean(low_qual ** 2))
        scale = rms_low_qual / rms_high_qual
        high_qual = scale * high_qual
        # Calculate spetra via FFT and plot in same figure
        boundary = 5
        mag_high_qual, f = plt.mlab.magnitude_spectrum(high_qual, Fs=sampling_rate)
        mag_high_qual_db = smooth(20 * np.log10(mag_high_qual), 14)
        plt.plot(
            f[boundary:-boundary],
            mag_high_qual_db[boundary:-boundary],
            color=red,
        )
        mag_low_qual, f = plt.mlab.magnitude_spectrum(low_qual, Fs=sampling_rate)
        mag_low_qual_db = smooth(20 * np.log10(mag_low_qual), 14)
        if low_qual_offset is not None:
            mag_low_qual_db += low_qual_offset
        plt.plot(
            f[boundary:-boundary],
            mag_low_qual_db[boundary:-boundary],
            color=blue,
        )
        plt.ylim([-128, -42])
        plt.ylabel('Magnitude / dB')
        plt.xlabel('Frequency / Hz')
        plt.grid(alpha=0.4)
        sns.despine()
        plt.tight_layout()
        

.. Plot spectrum
.. plot::
    :context: close-figs

    import audb
    import seaborn as sns


    sampling_rate = 16000

    # Load high quality example
    table = 'files'
    media = (
        'f1/'
        'dis-f1-b1.wav'
    )
    db = audb.load(
        'emovo',
        version='1.2.1',
        media=media,
        tables=table,
        sampling_rate=sampling_rate,
        format='wav',
        mixdown=True,
        verbose=False,
    )
    high_qual = load_audio(db[table])

    # Load low quality example
    table = 'files'
    media = (
        'converted/'
        'swb1_d1/data/'
        'sw02001.wav'
    )
    db = audb.load(
        'switchboard-1',
        version='1.0.0',
        media=media,
        tables=table,
        sampling_rate=sampling_rate,
        format='wav',
        mixdown=True,
        verbose=False,
    )
    low_qual = load_audio(db[table])

    plot_spectrum(high_qual, low_qual, sampling_rate, low_qual_offset=20)
    _ = plt.legend(['High Quality Sample', 'Low Quality Phone Sample'])


We mimic this behavior
by applying a dynamic range compressor
with a threshold of -20 dB,
a ratio of 0.8,
attack time of 0.01 s,
and a release time of 0.02 s
to the incoming high quality signal.
The outgoing signal
is then encoded
by the lossy Adaptive Multi-Rate (AMR) codec
with a bit rate of 7400
using its narrow band version
which involves a downsampling to 8000 Hz.
The signal is afterwards upsampled to 16000 Hz,
peak normalized,
and we add high pass filtered pink noise
with a gain of -25 dB.
The high pass employs a cutoff frequency of 3000 Hz
and an order of 2.
When applying the filters
we ensure that the overall signal level stays the same
if possible without clipping.


.. Prepare listening examples
.. plot::
    :context: close-figs

    import audeer
    import audiofile
    from common.robustness_low_quality_phone import low_quality_phone_transform

    out_dir = audeer.path(media_dir, 'robustness_low_quality_phone')
    audeer.mkdir(out_dir)

    speech_file = audeer.path(media_dir, 'speech.wav')
    speech, sampling_rate = audiofile.read(speech_file, always_2d=True)

    speech_low = low_quality_phone_transform(speech, sampling_rate)

    plot_spectrum(speech[0, :], speech_low[0, :], sampling_rate)
    _ = plt.legend(['Original Audio', 'Simulated Low Quality Phone'])

    # plt.ylim([-77.5, -47.5])

.. Provide listening examples
.. plot::
    :context: close-figs

    plot(speech, red, 'Original\nAudio')

.. raw:: html

    <p><audio controls src="media/speech.wav"></audio></p>

.. Lower Quality Phone
.. plot::
    :context: close-figs

    audiofile.write(
        audeer.path(out_dir, 'lower_quality_phone.wav'),
        speech_low,
        sampling_rate,
    )
    plot(speech_low, blue, 'Low Quality\nPhone')

.. raw:: html

    <p><audio controls src="media/robustness_low_quality_phone/lower_quality_phone.wav"></audio></p>


.. Extra space

|


We use the same definitions as in
:ref:`method-tests-robustness-small-changes`
to compute the difference :math:`\delta` in prediction.

The **Change CCC Low Quality Phone** test
ensures that the Concordance Correlation Coefficient (CCC)
does not decrease further
than by the given threshold
when applying
the low quality phone filter.

The **Change UAR Low Quality Phone**
tests ensure
that the Unweighted Average Recall (UAR)
does not decrease
too much when applying
the low quality phone filter.

The **Percentage Unchanged Predictions Low Quality Phone**
tests check
that the percentage of samples with
unchanged predictions is high enough
when applying
the low quality phone filter.
We use the same definitions as in the
:ref:`method-tests-robustness-small-changes`
to compute this percentage.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/robustness_low_quality_phone.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/robustness_low_quality_phone.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/robustness_low_quality_phone.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/robustness_low_quality_phone.csv


.. _switchboard-1: https://catalog.ldc.upenn.edu/LDC97S62


.. =======================================================================
.. _method-tests-robustness-recording-condition:

Robustness Recording Condition
------------------------------

The models should not change their
output when using a different microphone or
a different microphone position
to record the same audio.
To test this, we use databases that have
simultaneous recordings of the same audio
with different microphones and with different positions.


.. Headset
.. plot::
    :context: close-figs

    headset_file = audeer.path(media_dir, 'headset_speech.wav')
    headset, sampling_rate = audiofile.read(headset_file, always_2d=True)
    plot(headset, red, 'Original\nHeadset Audio')

.. raw:: html

    <p><audio controls src="media/headset_speech.wav"></audio></p>

.. Boundary
.. plot::
    :context: close-figs

    boundary_file = audeer.path(media_dir, 'boundary_speech.wav')
    boundary, sampling_rate = audiofile.read(boundary_file, always_2d=True)
    plot(boundary, blue, 'Boundary\nMicrophone')

.. raw:: html

    <p><audio controls src="media/boundary_speech.wav"></audio></p>

.. Mobile
.. plot::
    :context: close-figs

    mobile_file = audeer.path(media_dir, 'mobile_speech.wav')
    mobile, sampling_rate = audiofile.read(mobile_file, always_2d=True)
    plot(mobile, blue, 'Mobile\nMicrophone')

.. raw:: html

    <p><audio controls src="media/mobile_speech.wav"></audio></p>


.. Extra space

|

The **Percentage Unchanged Predictions Recording Condition**
test compares the prediction
on the audio in one recording condition
to the same audio in another recording
condition, and checks
that the percentage of unchanged predictions
is high enough.
We use the same definitions as in the
:ref:`method-tests-robustness-small-changes`
to compute this percentage.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/robustness_recording_condition.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/robustness_recording_condition.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/robustness_recording_condition.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/robustness_recording_condition.csv


.. =======================================================================
.. _method-tests-robustness-simulated-recording-condition:

Robustness Simulated Recording Condition
----------------------------------------

As described in the :ref:`method-tests-robustness-recording-condition` test,
the models should give the same or at least a similar output
for the same audio but recorded in different conditions.
To expand on the :ref:`method-tests-robustness-recording-condition` test
we simulate different recording conditions in this test.

We augment clean speech samples with
impulse responses corresponding to
different audio locations from the mardy_ database :cite:`Wen2006`
as well as impulse responses 
corresponding to different rooms
from the air_ database :cite:`Jeub2009`.
For the position test we use the impulse response
in the center position at 1 meter distance as the base (or reference)
position to compare all other positions to.
For the room test we use the impulse response of a
recording booth and compare to impulse responses of other rooms
recorded at similar distances as the reference impulse response.


.. Speech
.. plot::
    :context: close-figs

    from common.robustness_simulated_recording_condition import ir_transform

    out_dir = audeer.path(media_dir, 'robustness_simulated_recording_condition')
    audeer.mkdir(out_dir)

    speech_file = audeer.path(media_dir, 'headset_speech.wav')
    speech, sampling_rate = audiofile.read(speech_file, always_2d=True)
    plot(speech, red, 'Original\nAudio')

.. raw:: html

    <p><audio controls src="media/headset_speech.wav"></audio></p>

.. Base Position
.. plot::
    :context: close-figs

    base_position = ir_transform(
        speech,
        sampling_rate,
        ir_type='position',
        reference=True,
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'base_position.wav'),
        base_position,
        sampling_rate,
    )
    plot(base_position, blue, 'Base\nPosition')

.. raw:: html

    <p><audio controls src="media/robustness_simulated_recording_condition/base_position.wav"></audio></p>

.. Diff Position
.. plot::
    :context: close-figs

    diff_position = ir_transform(
        speech,
        sampling_rate,
        ir_type='position',
        reference=False,
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'diff_position.wav'),
        diff_position,
        sampling_rate,
    )
    plot(diff_position, blue, 'Diff.\nPosition')

.. raw:: html

    <p><audio controls src="media/robustness_simulated_recording_condition/diff_position.wav"></audio></p>

.. Base Room
.. plot::
    :context: close-figs

    base_room = ir_transform(
        speech,
        sampling_rate,
        ir_type='location', 
        reference=True,
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'base_room.wav'),
        base_room,
        sampling_rate,
    )
    plot(base_room, blue, 'Base\nRoom')

.. raw:: html

    <p><audio controls src="media/robustness_simulated_recording_condition/base_room.wav"></audio></p>


.. Diff Room
.. plot::
    :context: close-figs

    diff_room = ir_transform(
        speech,
        sampling_rate,
        ir_type='location', 
        reference=False,
        demo=True,
    )
    audiofile.write(
        audeer.path(out_dir, 'diff_room.wav'),
        diff_room,
        sampling_rate,
    )
    plot(diff_room, blue, 'Diff.\nRoom')

.. raw:: html

    <p><audio controls src="media/robustness_simulated_recording_condition/diff_room.wav"></audio></p>


.. Extra space

|

The **Percentage Unchanged Predictions Simulated Position**
test compares the prediction
on the audio with a simulated base position
to that of
the same audio with
a different simulated position,
and checks
that the percentage of samples with
unchanged predictions is high enough.
We use the same definitions as in the
:ref:`method-tests-robustness-small-changes`
to compute this percentage.

The **Percentage Unchanged Predictions Simulated Room**
test compares the prediction
on the audio with a simulated base room
to that of
the same audio with
a different simulated room,
and checks
that the percentage of samples with
unchanged predictions is high enough.
We use the same definitions as in the
:ref:`method-tests-robustness-small-changes`
to compute this percentage.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/robustness_simulated_recording_condition.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/robustness_simulated_recording_condition.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/robustness_simulated_recording_condition.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/robustness_simulated_recording_condition.csv

.. _mardy: https://www.commsp.ee.ic.ac.uk/~sap/resources/mardy-multichannel-acoustic-reverberation-database-at-york-database/
.. _air: https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/


.. =======================================================================
.. _method-tests-robustness-small-changes:

Robustness Small Changes
------------------------

The models should not change their output
if we apply very small changes to the input signals.
To test this we apply small changes
to the input signal and compare the predictions.
For regression, we calculate
the difference :math:`\delta_\text{reg}`
in prediction

.. math::

    \delta_\text{reg}(\text{segment}_s) =
      \|
        \text{prediction}_\text{reg}(\text{segment}_s)
        - \text{prediction}_\text{reg}(\text{augment}(\text{segment}_s))
      \|

for each segment :math:`\text{segment}_s`,
with :math:`\text{prediction}_\text{reg}(\cdot) \in [0, 1]`.
The percentage of unchanged predictions
for regression is then
given by the percentage
of all segments :math:`S`
with :math:`\delta_\text{reg} < 0.05`:

.. math::

    \text{percentage\_unchanged}_\text{reg} =
        \frac{
          |
            \{\text{segment}_s \; | \; \delta_\text{reg}(\text{segment}_s) < 0.05 \;
              \text{and} \; 1 \leq s \leq S\}
          |
        }{
          | \{\text{segment}_s \; | \; 1 \leq s \leq S\} |
        }

For classification
the difference :math:`\delta_\text{cls}`
in prediction is given by

.. math::

    \delta_\text{cls}(\text{segment}_s) =
      \|
        \text{prediction}_\text{cls}(\text{segment}_s)
        - \text{prediction}_\text{cls}(\text{augment}(\text{segment}_s))
      \|_2

for each segment :math:`\text{segment}_s`,
with :math:`\text{prediction}_\text{cls}(\cdot) \in \{\mathbf{e}^{(i)} \; | \; 1 \leq i \leq C\}`,
where :math:`\mathbf{e}^{(i)}`
is a one-hot vector corresponding to
one of the :math:`C` classes.
The percentage of unchanged predictions
for classification is then
given by the percentage
of all segments :math:`S`
with :math:`\delta_\text{cls} = 0`:

.. math::

    \text{percentage\_unchanged}_\text{cls} =
        \frac{
          |
            \{\text{segment}_s \; | \; \delta_\text{cls}(\text{segment}_s) = 0 \;
              \text{and} \; 1 \leq s \leq S\}
          |
        }{
          | \{\text{segment}_s \; | \; 1 \leq s \leq S\} |
        }

All the changes we apply here,
were optimized by listening
to one example augmented audio file
and adjusting the settings
so that a user perceives the changes
as subtle.

The **Percentage Unchanged Predictions Additive Tone** test
adds a sinusoid
with a frequency randomly selected
between 5000 Hz and 7000 Hz,
with a peak based
signal-to-noise ratio
randomly selected
from 40 dB, 45 dB, 50 dB
and checks that the percentage of unchanged
predictions is above the given threshold.

The **Percentage Unchanged Predictions Append Zeros** test
adds samples containing zeros
at the end of the input signal
and checks that the percentage of unchanged
predictions is above the given threshold.
The number of samples is randomly selected from
100, 500, 1000.

The **Percentage Unchanged Predictions Clip** test
clips a given percentage
of the input signal
and checks that the percentage of unchanged
predictions is above the given threshold.
The clipping percentage is randomly selected from
0.1%, 0.2%, 0.3%

The **Percentage Unchanged Predictions Crop Beginning** test
removes samples from the beginning
of an input signal
and checks that the percentage of unchanged
predictions is above the given threshold.
The number of samples is randomly selected from
100, 500, 1000.

The **Percentage Unchanged Predictions Crop End** test
removes samples from the end
of an input signal
and checks that the percentage of unchanged
predictions is above the given threshold.
The number of samples is randomly selected from
100, 500, 1000.

The **Percentage Unchanged Predictions Gain** test
changes the gain of an input signal
by a value randomly selected from
-2 dB, -1 dB, 1 dB, 2 dB
and checks that the percentage of unchanged
predictions is above the given threshold.

The **Percentage Unchanged Predictions Highpass Filter** test
applies a high pass Butterworth filter
of order 1
to the input signal
with a cutoff frequency randomly selected from
50 Hz, 100 Hz, 150 Hz
and checks that the percentage of unchanged
predictions is above the given threshold.

The **Percentage Unchanged Predictions Lowpass Filter** test
applies a low pass Butterworth filter
of order 1
to the input signal
with a cutoff frequency randomly selected from
7500 Hz, 7000 Hz, 6500 Hz
and checks that the percentage of unchanged
predictions is above the given threshold.

The **Percentage Unchanged Predictions Prepend Zeros** test
adds samples containing zeros
at the beginning of the input signal
and checks that the percentage of unchanged
predictions is above the given threshold.
The number of samples is randomly selected from
100, 500, 1000.

The **Percentage Unchanged Predictions White Noise** test
adds Gaussian distributed white noise
to the input signal
with a root mean square based
signal-to-noise ratio
randomly selected from
35 dB, 40 dB, 45 dB
and checks that the percentage of unchanged
predictions is above the given threshold.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/robustness_small_changes.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/robustness_small_changes.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/robustness_small_changes.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/robustness_small_changes.csv


.. =======================================================================
.. _method-tests-robustness-spectral-tilt:

Robustness Spectral Tilt
------------------------

The models should be robust
against boosting low or high frequencies
in the spectrum.
We simulate such spectral tilts
by attenuating or emphasizing the signal linearly.
This is achieved
by convolving the signal
with appropriate filters
as shown in the figure below.

.. Plot impulse responses and prepare listening examples
.. plot::
    :context: close-figs

    from common.robustness_spectral_tilt import spectral_tilt_transform

    speech_file = audeer.path(media_dir, 'speech.wav')
    speech, sampling_rate = audiofile.read(speech_file, always_2d=True)

    ir_file_downward = audeer.path(
        media_dir,
        'impulse_response_tilt_down_gain_0.0.wav',
    )
    ir_file_upward = audeer.path(
        media_dir,
        'impulse_response_tilt_up_gain_3.0.wav',
    )

    ir_downward, fs = audiofile.read(ir_file_downward)
    _ = plt.magnitude_spectrum(ir_downward, Fs=fs, scale="dB")

    ir_upward, fs = audiofile.read(ir_file_upward)
    ir_upward = 0.5 * ir_upward
    _ = plt.magnitude_spectrum(ir_upward, Fs=fs, scale="dB")

    plt.ylim([-77.5, -47.5])
    plt.legend(['Downward Spectral Tilt', 'Upward Spectral Tilt'])
    plt.ylabel('Magnitude / dB')
    plt.xlabel('Frequency / Hz')
    plt.grid(alpha=0.4)

    sns.despine()
    plt.tight_layout()

When applying the filters
we ensure that the overall signal level stays the same
if possible without clipping.

.. Speech
.. plot::
    :context: close-figs

    out_dir = audeer.path(media_dir, 'robustness_spectral_tilt')
    audeer.mkdir(out_dir)

    speech_file = audeer.path(media_dir, 'speech.wav')
    speech, sampling_rate = audiofile.read(speech_file, always_2d=True)
    plot(speech, red, 'Original\nAudio')

.. raw:: html

    <p><audio controls src="media/speech.wav"></audio></p>

.. Downward Tilt
.. plot::
    :context: close-figs

    downward_tilt = spectral_tilt_transform(
        speech,
        sampling_rate,
        ir_file_downward,
    )
    audiofile.write(
        audeer.path(out_dir, 'downward_tilt.wav'),
        downward_tilt,
        sampling_rate,
    )
    plot(downward_tilt, blue, 'Downward\nTilt')

.. raw:: html

    <p><audio controls src="media/robustness_spectral_tilt/downward_tilt.wav"></audio></p>

.. Upward Tilt
.. plot::
    :context: close-figs

    upward_tilt = spectral_tilt_transform(
        speech,
        sampling_rate,
        ir_file_upward,
    )
    audiofile.write(
        audeer.path(out_dir, 'upward_tilt.wav'),
        upward_tilt,
        sampling_rate,
    )
    plot(upward_tilt, blue, 'Upward\nTilt')

.. raw:: html

    <p><audio controls src="media/robustness_spectral_tilt/upward_tilt.wav"></audio></p>


.. Extra space

|


The **Change CCC Downward Tilt**
and **Change CCC Upward Tilt** tests
ensure that the Concordance Correlation Coefficient (CCC)
does not decrease
too much when applying
the downward or upward spectral tilt filter.

The **Change UAR Downward Tilt**
and **Change UAR Upward Tilt** tests
ensure that the Unweighted Average Recall (UAR)
does not decrease
too much when applying
the downward or upward spectral tilt filter.

The **Percentage Unchanged Predictions Downward Tilt**
and **Percentage Unchanged Predictions Upward Tilt**
tests check
that the percentage of samples with
unchanged predictions is high enough
when applying
the downward or upward spectral tilt filter.
We use the same definitions as in the
:ref:`method-tests-robustness-small-changes`
to compute this percentage.

.. csv-table:: Overview of tests and thresholds for arousal
    :header-rows: 1
    :file: method-tests/arousal/robustness_spectral_tilt.csv

.. csv-table:: Overview of tests and thresholds for dominance
    :header-rows: 1
    :file: method-tests/dominance/robustness_spectral_tilt.csv

.. csv-table:: Overview of tests and thresholds for valence
    :header-rows: 1
    :file: method-tests/valence/robustness_spectral_tilt.csv

.. csv-table:: Overview of tests and thresholds for emotion
    :header-rows: 1
    :file: method-tests/emotion/robustness_spectral_tilt.csv


.. =======================================================================
.. Links
.. _Jensen-Shannon divergence: https://en.wikipedia.org/wiki/Jensen–Shannon_divergence
.. _praat: http://www.praat.org/
.. _Spearman's rank correlation coefficient: https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient
