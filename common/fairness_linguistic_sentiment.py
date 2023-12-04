r"""Test fairness of linguistic sentiment.

The predictons returned by a model
should be equally dependent to
text sentiment for all languages
in which we would like to apply the model.
"""
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
    bin_proportion_shift_difference,
    class_proportion_shift_difference,
    mean_shift_difference,
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

    # Select maximum number of samples as some tables may have
    # a large number of samples
    max_num_samples = 2000

    sampling_rate = 16000
    mixdown = True
    format = 'wav'

    results = []
    for database in databases:
        name = database['name']
        version = database['version']
        table = database['table']
        language = database['language']
        db = audb.load(
            name,
            version=version,
            tables=[table],
            sampling_rate=sampling_rate,
            mixdown=mixdown,
            format=format,
            verbose=False,
            num_workers=4,
        )
        df_all = db[table].get()
        # Randomize df order,
        # Then select up to max_num_samples per language
        df_all = df_all.sample(frac=1, random_state=1).groupby('language').head(max_num_samples)
        prediction_all = run_model(model, df_all.index)
        prediction_all = prediction_all[condition]
        prediction_all.index = df_all.index

        task = f'{name}-{version}-{table}-{language}'

        result = test_metrics(
            metrics,
            task,
            prediction_all,
            df_all['sentiment'],
            df_all['language'],
            language,
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

    # Minimum number of samples required per bin for 1000 samples per group
    # calculated as CDF[NormalDistribution[0.5, 1/6], 0.25] * 1000
    # (Expected number of samples in the range of [-inf, 0.25] for a gaussian
    # distribution with mean=0.5 and std=1/6, when sampling 1000 times)
    min_samples = 67

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
        metric['normalize'] = True

        if 'Positive' in metric['name']:
            metric['sentiment'] = 'positive'
        elif 'Negative' in metric['name']:
            metric['sentiment'] = 'negative'
        elif 'Neutral' in metric['name']:
            metric['sentiment'] = 'neutral'

        if 'Mean Shift Difference' in metric['name']:
            metric['function'] = mean_shift_difference
            metric['comparison'] = abs_lt_or_nan

        elif 'Bin Proportion Shift Difference' in metric['name']:
            metric['function'] = (
                lambda predictions, language, language_value, sentiment, sentiment_value:
                bin_proportion_shift_difference(
                    predictions, language, language_value, sentiment, sentiment_value,
                    bins=4, min_samples=min_samples
                )
            )
            metric['comparison'] = abs_lt_or_nan

        elif 'Class Proportion Shift Difference' in metric['name']:
            metric['function'] = (
                lambda predictions, language, language_value, sentiment, sentiment_value:
                class_proportion_shift_difference(
                    predictions, language, language_value, sentiment, sentiment_value,
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
        predictions: typing.Union[typing.Sequence, pd.Series],
        sentiment: typing.Union[typing.Sequence, pd.Series],
        language: typing.Union[typing.Sequence, pd.Series],
        language_value: str,
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
            truth_label = 'Prediction all languages'
            prediction_label = f'Prediction {task.split("-")[-1]}'
            class_labels = CATEGORY_LABELS[model.condition] \
                if model.mode == 'classification' else None
            bins = None if model.mode == 'classification' \
                else list(np.arange(0, 1, 0.03))
            result = metric['function'](
                predictions,
                predictions.loc[language==language_value],
                truth_groups=sentiment,
                prediction_groups=sentiment.loc[language==language_value],
                group_order=['negative', 'neutral', 'positive'],
                order=class_labels,
                truth_label=truth_label,
                prediction_label=prediction_label,
                bins=bins
            )
        else:
            group_ref, avg_ref, result = metric['function'](
                predictions, language, language_value, sentiment, metric['sentiment']
            )

        # Store plot if metric has one
        if metric['plot']:
            name = metric['name'].lower().replace(' ', '-')
            result_file = os.path.join(
                result_dir,
                figure_filename(name, task),
            )
            plt.suptitle(task)
            plt.tight_layout()
            plt.savefig(result_file)
            plt.close()
        # Continue if we have only the plot
        if result != 0 and not result:
            continue

        if not isinstance(result, dict):
            result = {'': result}
            group_ref = {'': group_ref}
            avg_ref = {'': avg_ref}

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
                # Store results to dataframe to save later as CSV
                df[column_name] = score
                df[f'_{column_name}'] = ''
            else:
                # Round to ensure reproducibility of results
                score = np.round(score, 2)
                passed = metric['comparison'](score, threshold)
                score = metric['display'](score)
                # Store results to dataframe to save later as CSV
                df[column_name] = score
                # Store annotation values of group shift and average shift
                df[f'_{column_name}'] = (
                    f'({group_ref[class_label]:.2f} - '
                    f'{avg_ref[class_label]:.2f})'
                )

            print_result(task, display_name, score, passed)

        results[metric['name']] = df

    return results
