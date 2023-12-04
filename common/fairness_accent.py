r"""Test fairness of accent.

The predictons returned by a model
should be more or less independent
of the spoken accent.
"""
import operator
import os
import typing
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audb
import audeer
import audonnx

from . import (
    figure_filename,
    model_name,
    print_header,
    print_result,
    run_model,
    CATEGORY_LABELS,
)
from common.comparisons import abs_lt_or_nan
from common.metrics import (
    difference_in_mean,
    relative_difference_per_bin,
    relative_difference_per_class,
)
from common.plots import plot_distribution_normalized


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TEST = audeer.basename_wo_ext(os.path.abspath(__file__))
TEST_NAME = audeer.basename_wo_ext(__file__)


def run(
        model: audonnx.Model,
        condition: str,
):
    r"""Execute tests.

    Args:
        model: model to run
        condition: test condition, e.g. ``'arousal'``

    """
    if model.uid == 'random-rater':
        return 1

    print_header(TEST_NAME)

    # Load metrics and data to test from config file
    databases, metrics = load_config(condition)

    result_dir = os.path.join(
        CURRENT_DIR,
        '..',
        'docs',
        'results',
        'test',
        condition,
        model_name(model),
        CURRENT_TEST,
    )
    audeer.rmdir(result_dir)
    audeer.mkdir(result_dir)

    sampling_rate = 16000
    mixdown = True
    format = 'wav'

    # Load truth from English test set.
    db = audb.load(
        'speech-accent-archive',
        version='2.2.0',
        tables=['accent.test'],
        sampling_rate=sampling_rate,
        mixdown=mixdown,
        format=format,
        verbose=False,
        num_workers=4,
    )
    df_all = db['accent.test'].get()
    prediction_all = run_model(model, df_all.index)
    prediction_all = prediction_all[condition]
    prediction_all.index = df_all.index

    # Calculate predictions for all languages
    results = []
    for database in databases:

        name = database['name']
        version = database['version']
        split = database['split']

        if name != 'speech-accent-archive' or version != '2.2.0':
            raise ValueError('Only speech-accent-archive v2.2.0 is supported')

        df = df_all[df_all.native_language == split]

        # Compare the prediction of the current native language to all native languages
        prediction = prediction_all.loc[df.index]
        if split == 'english':
            task = f'{name}-{version}-native'
        else:
            task = f'{name}-{version}-{split}'


        task = f'{name}-{version}-{split}'
        result = test_metrics(
            metrics,
            task,
            prediction_all,
            prediction,
            result_dir,
            model,
        )

        results.append(result)

    # Store results to docs/ folder
    for metric in metrics:
        name = metric['name']
        # Skip if metric has not added any results
        if name not in results[0]:
            continue
        metric_result = pd.concat([result[name] for result in results])
        filename = name.lower().replace(' ', '-')
        result_file = os.path.join(result_dir, f'{filename}.csv')
        metric_result.to_csv(result_file)


def load_config(condition):
    r"""Load configuration from YAML file.

    This loads the settings from a YAML file
    with the same basename as this script.

    The settings include the databases
    to use with the following dictionary entries:
    * 'name'
    * 'version'
    * 'split'

    In addition, test metrics are loaded
    and corresponding functions
    and test criteria are added.
    The returned metrics list will contain
    dictionaries for every metric
    with some of the following content:
    * 'name' - name of the metric displayed in the test output
    * 'short' - short name of metric used in result table
    * 'function' - metric assuming the two input args truth and prediction
    * 'threshold' - threshold of calculated metric to pass the test
    * 'comparison' - comparison assuming score and threshold as input args
    * 'display' - function to convert score value to string
    * 'plot' - if the metric is returning an image instead of a value

    Args:
        condition: test condition, e.g. ``'arousal'``

    Returns:
        * databases
        * metrics

    """

    # Minimum number of samples required per bin for 60 samples per group
    # calculated as CDF[NormalDistribution[0.5, 1/6], 0.25] * 60
    # (Expected number of samples in the range of [-inf, 0.25] for a gaussian
    # distribution with mean=0.5 and std=1/6, when sampling 60 times)
    min_samples = 4

    yaml_file = os.path.join(
        CURRENT_DIR,
        '..',
        'test',
        condition,
        f'{TEST_NAME}.yaml',
    )
    with open(yaml_file, 'r') as fp:
        d = yaml.safe_load(fp)
        metrics = d['Metrics']
        databases = d['Data']

    for metric in metrics:
        metric['plot'] = False

        if metric['name'] == 'Mean Value':
            metric['function'] = difference_in_mean
            metric['comparison'] = abs_lt_or_nan
        elif metric['name'] == 'Relative Difference Per Bin':
            metric['function'] = (
                lambda truth, prediction:
                relative_difference_per_bin(truth, prediction, bins=4,
                                            min_samples=min_samples)
            )
            metric['comparison'] = abs_lt_or_nan

        elif metric['name'] == 'Relative Difference Per Class':
            metric['function'] = (
                lambda truth, prediction:
                relative_difference_per_class(truth, prediction,
                    CATEGORY_LABELS[condition]
                )
            )
            metric['comparison'] = abs_lt_or_nan

        elif metric['name'] == 'Visualization':
            metric['function'] = plot_distribution_normalized
            metric['plot'] = True

        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        task: str,
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        result_dir: str,
        model: audonnx.Model,
) -> typing.Dict:
    r"""Execute calculation of metrics.

    If the metric is a plot,
    it will store the plot file
    in ``result_dir``.

    Otherwise it will return the metric calculation
    as a dataframe
    using the task as index entry
    within a dictionary,
    named after the metric.

    """
    results = {}
    for metric in metrics:

        df = pd.DataFrame(index=[task])
        df.index.name = 'Data'

        if metric['plot']:
            truth_label = 'Prediction combined'
            prediction_label = f'Prediction {task.split("-")[-1]}-english'
            if model.mode == 'classification':
                order = CATEGORY_LABELS[model.condition]
                bins = None
            else:
                order = None
                bins = list(np.arange(0, 1, 0.03))
            result = metric['function'](
                truth,
                prediction,
                order,
                truth_label=truth_label,
                prediction_label=prediction_label,
                bins=bins,
            )

        else:
            # Run metric function to get scores
            result = metric['function'](truth, prediction)

        # Store plot if metric has one
        if metric['plot']:
            name = metric['name'].lower().replace(' ', '-')
            result_file = os.path.join(
                result_dir,
                figure_filename(name, task),
            )
            plt.title(task)
            plt.tight_layout()
            plt.savefig(result_file)
            plt.close()
        # Continue if we have only the plot
        if result != 0 and not result:
            continue

        if not isinstance(result, dict):
            result = {'': result}

        for class_label, score in result.items():
            if class_label != '':
                column_name = class_label
                display_name = f'{metric["short"]} {column_name}'
            else:
                column_name = metric['name']
                display_name = metric['name']

            threshold = metric['threshold']
            if score is None:
                passed = None
            else:
                # Round to ensure reproducibility of results
                score = np.round(score, 2)
                passed = metric['comparison'](score, threshold)
                score = metric['display'](score)

            # Store results to dataframe to save later as CSV
            df[column_name] = score
            print_result(task, display_name, score, passed)

        results[metric['name']] = df

    return results
