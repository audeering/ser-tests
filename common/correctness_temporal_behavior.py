r"""Test correctness of temporal behvaior.

The predictons returned by a model
should match the temporal behavior (value toggling)
as closely as possible over a high number of samples.
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
import audmetric
import audonnx

from . import (
    CATEGORY_LABELS,
    figure_filename,
    model_name,
    print_header,
    print_result,
    run_model,
)
from common.comparisons import abs_lt
from common.metrics import (
    jensen_shannon_distance,
    mean_directional_error,
    reactivity,
    stability,
    value_changes_per_file,
)
from common.plots import plot_distribution_value_changes


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

    results = []
    for database in databases:

        name = database['name']
        version = database['version']
        table = database['table']
        column = database['column']

        db = audb.load(
            name,
            version=version,
            tables=[table],
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
        )
        truth = db[table][column].get()
        # Filter for relevant available class labels
        if condition in CATEGORY_LABELS:
            class_labels = CATEGORY_LABELS[condition]
            truth = truth[truth.isin(class_labels)]
        prediction = run_model(model, truth.index, database)
        prediction = prediction[condition]

        # Filter for relevant available labels
        truth = truth.dropna()
        prediction = prediction.loc[truth.index]

        # We are not interested here in the values,
        # but the number of changes per file
        truth = value_changes_per_file(truth)
        prediction = value_changes_per_file(prediction)


        task = f'{name}-{version}-{table}'
        result = test_metrics(
            metrics,
            task,
            truth,
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

    # Define binning of tone changes per segment distributions
    if condition in CATEGORY_LABELS:
        # for classification we have the proportion of samples that
        # change from the previous segment, which could be between 0 and 1
        bins = list(np.arange(0, 1.1, 0.1))

        max_value = 1
    else:
        # for regression, we have the average value change per segment
        # across the wholerecording
        bins = list(np.arange(0, .31, 0.03))
        max_value = 0.3

    for metric in metrics:

        metric['plot'] = False

        if metric['name'] == 'Jensen-Shannon Distance':
            metric['function'] = (
                lambda truth, prediction:
                jensen_shannon_distance(truth, prediction, bins=bins)
            )
            metric['comparison'] = operator.lt
        elif metric['name'] == 'Mean Absolute Error':
            metric['function'] = audmetric.mean_absolute_error
            metric['comparison'] = abs_lt
        elif metric['name'] == 'Mean Directional Error':
            metric['function'] = mean_directional_error
            metric['comparison'] = abs_lt
        elif metric['name'] == 'Reactivity':
            metric['function'] = (
                lambda truth, prediction:
                reactivity(truth, prediction, max_value=max_value)
            )
            metric['comparison'] = operator.gt
        elif metric['name'] == 'Stability':
            metric['function'] = (
                lambda truth, prediction:
                stability(truth, prediction, max_value=max_value)
            )
            metric['comparison'] = operator.gt
        elif metric['name'] == 'Visualization':
            metric['function'] = (
                lambda truth, prediction:
                plot_distribution_value_changes(truth, prediction, bins)
            )
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
    if model.mode != 'classification':
        truth = truth.astype('float32')

    results = {}
    for metric in metrics:

        df = pd.DataFrame(index=[task])
        df.index.name = 'Data'
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
        if not result:
            continue

        # Fix numerical unstability when rerunning test
        result = np.round(result, 3)

        column_name = metric['name']
        display_name = metric['name']
        # Store results to dataframe to save later as CSV
        df[column_name] = result
        threshold = metric['threshold']
        passed = metric['comparison'](result, threshold)
        display_score = metric['display'](result)
        print_result(task, display_name, display_score, passed)

        results[metric['name']] = df



    return results
