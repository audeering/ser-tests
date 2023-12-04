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
    figure_filename,
    model_name,
    print_header,
    print_result,
    run_model,
    split_figure_title,
    CATEGORY_LABELS,
)
from common.plots import plot_confusion_matrix


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TEST = audeer.basename_wo_ext(os.path.abspath(__file__))
TEST_NAME = audeer.basename_wo_ext(__file__)


def run(
        model: audonnx.Model,
        condition: str,
):
    r"""Execute actual correctness tests.

    Args:
        model: model to run
        condition: test condition, e.g.``'emotion'``

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

    class_labels = CATEGORY_LABELS[condition]

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
        truth = truth[truth.isin(class_labels)]
        prediction = run_model(model, truth.index, database)
        # restore index back to filewise in case truth.index is filewise and the prediction index has changed
        prediction.index = truth.index
        prediction = prediction[condition]

        task = f'{name}-{version}-{table}'
        result = test_metrics(
            metrics,
            task,
            truth,
            prediction,
            class_labels,
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

        if metric['name'] == 'Unweighted Average Recall':
            metric['function'] = audmetric.unweighted_average_recall
            metric['comparison'] = operator.gt

        elif metric['name'] == 'Recall per Class':
            metric['function'] = audmetric.recall_per_class
            metric['comparison'] = operator.gt

        elif metric['name'] == 'Unweighted Average Precision':
            metric['function'] = audmetric.unweighted_average_precision
            metric['comparison'] = operator.gt

        elif metric['name'] == 'Precision per Class':
            metric['function'] = audmetric.precision_per_class
            metric['comparison'] = operator.gt

        elif metric['name'] == 'Mean Squared Error':
            metric['function'] = audmetric.mean_squared_error
            metric['comparison'] = operator.lt

        elif metric['name'] == 'Visualization':
            metric['function'] = plot_confusion_matrix
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
            # Use pd.concat to handle truth and prediction
            # with different length
            truth.name = 0
            prediction.name = 1
            combined = pd.concat([truth, prediction], axis=1)
            result = metric['function'](
                combined[0],
                combined[1],
                labels=list(class_labels),
            )
        else:
            result = metric['function'](truth, prediction)

        # Store plot if metric has one
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
