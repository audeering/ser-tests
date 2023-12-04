r"""Test robusteness small changes.

The model results should stay the same
for small changes to the input signal.
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
import auglib

from . import (
    model_name,
    print_header,
    print_result,
    run_model,
    CATEGORY_LABELS,
    split_figure_title,
)
from common.metrics import percentage_of_identity
from common.plots import plot_robustness


auglib.seed(0)
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
        truth = truth.dropna()
        if condition in CATEGORY_LABELS:
            class_labels = CATEGORY_LABELS[condition]
            truth = truth[truth.isin(class_labels)]

        task = f'{name}-{version}-{table}'
        result = test_metrics(
            metrics,
            condition,
            task,
            truth,
            model,
            result_dir,
        )

        results.append(result)

    # Store results to docs/ folder
    combined_robustness = []
    combined_results = []
    for metric in metrics:
        name = metric['name']

        # Store name for plot
        if metric['plot']:
            plot_name = name.lower()

        # Skip if metric has not added any results
        if name not in results[0]:
            continue

        # Store robustness values
        robustness = pd.concat([result[name] for result in results])
        combined_robustness.append(robustness)
        # Add test result ('passed' vs. 'failed')
        threshold = metric['threshold']

        # Gather results for creating summary plot.
        # This needs to store short names
        # and if a test has failed or not
        result = robustness.copy()
        if 'short' in metric:
            short_name = metric['short']
        else:
            short_name = name
        result = result[short_name].map(
            lambda x:
            'passed' if x > threshold
            else 'failed'
        )
        combined_results.append(result)

        filename = name.lower().replace(' ', '-')
        result_file = os.path.join(result_dir, f'{filename}.csv')
        robustness.to_csv(result_file)

    # Create overview plot
    df_robustness = pd.concat(combined_robustness, axis=1)
    df_robustness = df_robustness.sort_index(axis=1)
    df_result = pd.concat(combined_results, axis=1)
    df_result = df_result.sort_index(axis=1)
    for n in range(len(df_robustness)):

        # Create a data frame containing
        # test name, robustness, test result
        # as columns
        robustness = df_robustness.iloc[n]
        database = str(robustness.name)
        robustness.name = 'robustness'
        result = df_result.iloc[n]
        result.name = 'result'
        index = pd.Series(list(df_robustness.columns), name='index')
        index.index = robustness.index
        df = pd.concat([index, robustness, result], axis=1)
        title = split_figure_title(database, max_length=60)
        plot_robustness(df, title,
                        robustness_name='Percent Unchanged Pred')
        plt.tight_layout()

        outfile = os.path.join(result_dir, f'{plot_name}_{database}.png')
        plt.savefig(outfile)
        plt.close()


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
        condition: test condition, e.g.``'arousal'``

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
    for metric in metrics:

        if metric['name'] == 'Visualization':
            metric['plot'] = True
            continue

        if metric['name'] == 'Percentage Unchanged Predictions Additive Tone':
            transform = auglib.transform.Tone(
                freq=auglib.observe.IntUni(5000, 7000),
                snr_db=auglib.observe.List(
                    [40, 45, 50],
                    draw=True,
                ),
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Append Zeros':
            transform = auglib.transform.AppendValue(
                duration=auglib.observe.List(
                    [100, 500, 1000],
                    draw=True,
                ),
                unit='samples',
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Clip':
            transform = auglib.transform.ClipByRatio(
                ratio=auglib.observe.List(
                    [0.001, 0.002, 0.003],
                    draw=True,
                ),
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Crop Beginning':
            transform = auglib.transform.Trim(
                start_pos=auglib.observe.List(
                    [100, 500, 1000],
                    draw=True,
                ),
                unit='samples',
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Crop End':
            transform = auglib.transform.Trim(
                end_pos=auglib.observe.List(
                    [100, 500, 1000],
                    draw=True,
                ),
                unit='samples',
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Gain':
            transform = auglib.transform.GainStage(
                gain_db=auglib.observe.List(
                    [-2, -1, 1, 2],
                    draw=True,
                ),
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Highpass Filter':
            transform = auglib.transform.HighPass(
                cutoff=auglib.observe.List(
                    [50, 100, 150],
                    draw=True,
                ),
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Lowpass Filter':
            transform = auglib.transform.LowPass(
                cutoff=auglib.observe.List(
                    [7500, 7000, 6500],
                    draw=True,
                ),
            )

        elif metric['name'] == 'Percentage Unchanged Predictions Prepend Zeros':
            transform = auglib.transform.PrependValue(
                duration=auglib.observe.List(
                    [100, 500, 1000],
                    draw=True,
                ),
                unit='samples',
            )

        elif metric['name'] == 'Percentage Unchanged Predictions White Noise':
            transform = auglib.transform.WhiteNoiseGaussian(
                snr_db=auglib.observe.List(
                    [35, 40, 45],
                    draw=True,
                ),
            )

        metric['transform'] = transform
        metric['plot'] = False
        metric['function'] = percentage_of_identity
        metric['comparison'] = operator.gt
        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        condition: str,
        task: str,
        truth: typing.Union[typing.Sequence, pd.Series],
        model: audonnx.Model,
        result_dir: str,
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

        # Continue if only summary plot
        if metric['name'] == 'Visualization':
            continue

        # Apply augmentation and get metric scores
        augment = auglib.Augment(metric['transform'])
        truth_augmented = augment.augment(truth)
        prediction_truth = run_model(model, truth.index)
        prediction_truth = prediction_truth[condition]
        prediction_augmented = run_model(model, truth_augmented.index)
        prediction_augmented = prediction_augmented[condition]

        score = metric['function'](prediction_truth, prediction_augmented)

        # Continue if we have only the plot
        if not score and score != 0:
            continue

        if 'short' in metric:
            column_name = metric['short']
            display_name = metric['short']
        else:
            column_name = metric['name']
            display_name = metric['name']
        threshold = metric['threshold']

        # Round to ensure reproducibility of results
        score = np.round(score, 2)

        # Store results to dataframe to save later as CSV
        df[column_name] = score

        passed = metric['comparison'](score, threshold)
        display_score = metric['display'](score)
        print_result(task, display_name, display_score, passed)

        results[metric['name']] = df

    return results
