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
import audplot

from . import (
    figure_filename,
    model_name,
    print_header,
    print_result,
    run_model,
    split_figure_title,
)

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

    predictions = []
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
        prediction = run_model(model, truth.index, database)
        prediction.index = truth.index
        prediction = prediction[condition]

        # Filter for relevant available labels
        truth = truth.dropna()
        prediction = prediction.loc[truth.index]

        task = f'{name}-{version}-{table}'
        result = test_metrics(
            metrics,
            task,
            truth,
            prediction,
            result_dir,
        )

        results.append(result)
        predictions.append(prediction)

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
    * 'comparsion' - comparison assuming score and threshold as input args
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
    for metric in metrics:
        metric['plot'] = False

        if metric['name'] == 'Concordance Correlation Coeff':
            metric['function'] = audmetric.concordance_cc
            metric['comparison'] = operator.gt
            metric['display'] = lambda x: f'{x:.2f}'

        elif metric['name'] == 'Pearson Correlation Coeff':
            metric['function'] = audmetric.pearson_cc
            metric['comparison'] = operator.gt
            metric['display'] = lambda x: f'{x:.2f}'

        elif metric['name'] == 'Mean Absolute Error':
            metric['function'] = audmetric.mean_absolute_error
            metric['comparison'] = operator.lt
            metric['display'] = lambda x: f'{x:.3f}'

        elif metric['name'] == 'Visualization':
            metric['function'] = (
                lambda truth, prediction:
                audplot.scatter(
                    truth,
                    prediction,
                    fit=True,
                )
            )
            metric['plot'] = True

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        task: str,
        truth: typing.Union[typing.Sequence, pd.Series],
        prediction: typing.Union[typing.Sequence, pd.Series],
        result_dir: str,
) -> typing.Dict:
    r"""Execute calculation of metrics.

    If the metric is a plot,
    it will store the plot file
    in ``result_dir``.
    There will be one file per test and task.

    Otherwise it will return the metric calculation
    as a dataframe
    using the task as index entry
    within a dictionary,
    named after the metric.

    """
    truth = truth.astype('float32')

    results = {}
    for metric in metrics:

        df = pd.DataFrame(index=[task])
        df.index.name = 'Data'

        # Run metric function to get scores
        score = metric['function'](truth, prediction)

        # Store plot if metric has one
        if metric['plot']:
            name = metric['name'].lower().replace(' ', '-')
            # Add newline to task if too long
            title = split_figure_title(task, max_length=50)
            result_file = os.path.join(
                result_dir,
                figure_filename(name, task),
            )
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.title(title)
            fig = plt.gcf()
            fig.set_size_inches(4.8, 3.6)
            plt.tight_layout()
            fig.savefig(result_file)
            plt.close()

        # Continue if we have only the plot
        if not score and score != 0:
            continue

        column_name = metric['name']
        display_name = metric['name']

        # Round to ensure reproducibility of results
        score = np.round(score, 3)

        # Store results to dataframe to save later as CSV
        df[column_name] = score

        threshold = metric['threshold']
        passed = metric['comparison'](score, threshold)
        score = metric['display'](score)

        print_result(task, display_name, score, passed)

        results[metric['name']] = df

    return results
