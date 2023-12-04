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
    split_figure_title,
)
from common.comparisons import gt_or_nan
from common.metrics import proportion_in_range
from common.plots import plot_distribution_by_category


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TEST = audeer.basename_wo_ext(os.path.abspath(__file__))
TEST_NAME = audeer.basename_wo_ext(__file__)

# Certain emotions with unclear dimensional ranges are not included
CATEGORY_LOW = {
    "arousal": [
        "boredom",
        # "contempt",
        # "disgust",
        "sadness",
    ],
    "dominance": [
        "fear",
        "sadness",
    ],
    "valence": [
        "anger",
        # "contempt",
        "disgust",
        "fear",
        "frustration",
        "sadness",
    ]
}
CATEGORY_NEUTRAL = {
    "arousal": [
        "neutral",
    ],
    "dominance": [
        "happiness",
        "neutral",
        "surprise",
    ],
    "valence": [
        "boredom",
        "neutral",
    ]
}

CATEGORY_HIGH = {
    "arousal": [
        "anger",
        "fear",
        "surprise",
    ],
    "dominance": [
        "anger",
        # "contempt",
    ],
    "valence": [
        "happiness",
        # "surprise",
    ]
}


def run(
        model: audonnx.Model,
        condition: str,
):
    r"""Execute actual correctness tests.

    Args:
        model: model to run
        condition: test condition, e.g.``'arousal'``

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
        condition_labels = set(
            CATEGORY_LOW[condition] +
            CATEGORY_NEUTRAL[condition] +
            CATEGORY_HIGH[condition]
        )

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
        available_labels = set(truth.unique())
        labels = sorted(available_labels.intersection(condition_labels))
        truth = truth[truth.isin(labels)]
        prediction = run_model(model, truth.index, database)
        # restore index back to filewise in case truth.index is filewise and
        # the prediction index has changed
        prediction.index = truth.index
        prediction = prediction[condition]

        task = f'{name}-{version}-{table}'
        result = test_metrics(
            metrics,
            task,
            truth,
            prediction,
            labels,
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
        # sort columns alphabetically
        metric_result.sort_index(axis=1, inplace=True)
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
        condition: test condition, e.g. ``'emotion'``

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
        metric['plot'] = False

        if 'Samples in Expected' in metric['name']:
            metric['function'] = (
                lambda truth, prediction, labels, allowed_range:
                {
                    label: proportion_in_range(
                        prediction[truth == label], allowed_range)
                    for label in labels
                }
            )
            metric['comparison'] = gt_or_nan

        elif metric['name'] == 'Visualization':
            metric['function'] = (
                lambda truth, prediction, labels, expected_ranges:
                plot_distribution_by_category(
                    truth,
                    prediction,
                    order=labels,
                    expected_ranges=expected_ranges
                )
            )
            metric['plot'] = True

        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        task: str,
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        class_labels: typing.Sequence[str],
        result_dir: str,
        model: audonnx.Model,
) -> typing.Dict:
    r"""Execute calculation of metrics.

    If the metric is a plot,
    it will store the plot file
    in ``result_dir``.
    There will be one file per test and condition.

    Otherwise it will return the metric calculation
    as a dataframe
    using the condition as index entry
    within a dictionary,
    named after the metric.

    """
    results = {}
    for metric in metrics:

        df = pd.DataFrame(index=[task])
        df.index.name = 'Data'

        # Store plot if metric has one
        if metric['plot']:
            # get expected ranges for each category to draw in plot
            expected_ranges = {
                class_label: (None, 0.45) for class_label
                in CATEGORY_LOW[model.condition]
            }
            expected_ranges.update({
                class_label: (0.3, 0.7) for class_label
                in CATEGORY_NEUTRAL[model.condition]
            })
            expected_ranges.update({
                class_label: (0.55, None) for class_label
                in CATEGORY_HIGH[model.condition]
            })
            # Use pd.concat to handle truth and prediction
            # with different length
            truth.name = 0
            prediction.name = 1
            combined = pd.concat([truth, prediction], axis=1)
            result = metric['function'](
                combined[0],
                combined[1],
                list(class_labels),
                expected_ranges
            )
        else:
            if 'Low' in metric['name']:
                metric_labels = CATEGORY_LOW[model.condition]
                allowed_range = (None, 0.45)
            elif 'Neutral' in metric['name']:
                metric_labels = CATEGORY_NEUTRAL[model.condition]
                allowed_range = (0.3, 0.7)
            elif 'High' in metric['name']:
                metric_labels = CATEGORY_HIGH[model.condition]
                allowed_range = (0.55, None)
            else:
                raise ValueError(f'Unsupported metric {metric}')
            result = metric['function'](
                truth, prediction, set(metric_labels).intersection(
                    class_labels), allowed_range
            )

        if metric['plot']:
            name = metric['name'].lower().replace(' ', '-')
            # Add newline to task if too long
            title = split_figure_title(task, max_length=50)
            result_file = os.path.join(
                result_dir,
                figure_filename(name, task),
            )
            plt.title(title)
            plt.tight_layout()
            plt.savefig(result_file)
            plt.close()
        # Continue if we have only the plot
        if not result:
            continue

        # Force results to be stored as a dictionary
        # with class name as key,
        # defaulting to '' for all classes
        if not isinstance(result, dict):
            result = {'': result}

        for class_label, score in result.items():
            if class_label != '':
                column_name = class_label
                display_name = f'{metric["short"]} {column_name}'
            else:
                column_name = metric['name']
                display_name = metric['name']

            # Round to ensure reproducibility of results
            score = np.round(score, 2)

            # Store results to dataframe to save later as CSV
            df[column_name] = score

            if isinstance(metric['threshold'], dict):
                threshold = metric['threshold'][class_label]
            else:
                threshold = metric['threshold']

            passed = metric['comparison'](score, threshold)
            score = metric['display'](score)

            print_result(task, display_name, score, passed)

        results[metric['name']] = df

    return results
