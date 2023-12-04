r"""Test correctness of speaker average value.

If we average the value over all segments
of a speaker
it should not result in a different
average value as in the ground truth labels.
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
import audformat
import audmetric
import audonnx

from . import (
    anonymize_column,
    average_score,
    CATEGORY_LABELS,
    class_proportions,
    figure_filename,
    model_name,
    print_header,
    print_result,
    run_model,
)
from common.metrics import mean_directional_error
from common.comparisons import abs_lt
from common.plots import plot_average_value


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
        if 'tables' in database:
            tables = database['tables']
        else:
            tables = [database['table']]
        column = database['column']
        speaker_table = database['speaker_table']
        speaker_column = database['speaker_column']

        db = audb.load(
            name,
            version=version,
            tables=tables+[speaker_table],
            sampling_rate=16000,
            mixdown=True,
            format='wav',
            verbose=False,
            num_workers=4,
        )
        df = sum(
            [db[table] for table in tables],
            start=audformat.Table(),
        ).get()

        # Filter for relevant available class labels
        if condition in CATEGORY_LABELS:
            class_labels = CATEGORY_LABELS[condition]
            df = df[df[column].isin(class_labels)]
        y = run_model(model, df.index, database)
        # reset index in case file wise index is needed for comparison
        y.index = df.index

        # Filter for relevant available labels and available speaker labels
        df['speaker'] = db[speaker_table][speaker_column].get(index=df.index)
        df = df.dropna(subset=[column, 'speaker'])

        # Only include speakers with enough overall samples
        speaker_samples = df.groupby('speaker')[column].count()
        n_sample_threshold = 10
        speakers_enough_samples = set(speaker_samples[
            speaker_samples >= n_sample_threshold
        ].index.to_list())

        if condition in CATEGORY_LABELS:
            class_n_sample_threshold = 8
            # Only include speakers with enough samples per class
            for class_label in class_labels:
                speaker_class_samples = (
                    df[df[column] == class_label]
                    .groupby('speaker')[column]
                    .count()
                )
                speakers_enough_class_samples = set(speaker_class_samples[
                    speaker_class_samples >= class_n_sample_threshold
                ].index.to_list())
                speakers_enough_samples = speakers_enough_samples.intersection(
                    speakers_enough_class_samples)
        df = df[df['speaker'].isin(speakers_enough_samples)]

        # filter prediction to index with relevant samples
        y = y.loc[df.index]

        df = anonymize_column(df, speaker_column)
        df_prediction = pd.concat([y, df[speaker_column]], axis=1)

        speakers = list(df[speaker_column].unique())
        if condition in CATEGORY_LABELS:
            truth = pd.DataFrame(
                index=speakers,
                columns=CATEGORY_LABELS[condition],
                dtype='float32',
            )
            prediction = pd.DataFrame(
                index=speakers,
                columns=CATEGORY_LABELS[condition],
                dtype='float32',
            )
            for speaker in speakers:
                # for categories, count the percentage of each class
                truth.loc[speaker, :] = class_proportions(
                    df, speaker_column, speaker, column)
                prediction.loc[speaker, :] = class_proportions(
                    df_prediction, speaker_column, speaker, column)

        else:
            truth = pd.Series(
                index=speakers,
                name=column,
                dtype='float32',
            )
            prediction = pd.Series(
                index=speakers,
                name=column,
                dtype='float32',
            )
            for speaker in speakers:
                truth.loc[speaker] = average_score(
                    df, speaker_column, speaker, column
                )
                prediction.loc[speaker] = average_score(
                    df_prediction,
                    speaker_column,
                    speaker,
                    column,
                )

        table_name = database.get('table_alias', '-'.join(tables))
        task = f'{name}-{version}-{table_name}'

        result = test_metrics(
            metrics,
            task,
            truth,
            prediction,
            result_dir,
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
    for metric in metrics:
        metric['plot'] = False
        if 'Mean Absolute Error' in metric['name']:
            if 'Class Proportion' in metric['name']:
                metric['function'] = (
                    lambda truth, prediction: {
                        class_label: audmetric.mean_absolute_error(
                            truth[class_label], prediction[class_label])
                        for class_label in truth.columns
                    }
                )
            else:
                metric['function'] = audmetric.mean_absolute_error
            metric['comparison'] = operator.lt
        elif 'Mean Directional Error' in metric['name']:
            if 'Class Proportion' in metric['name']:
                metric['function'] = (
                    lambda truth, prediction: {
                        class_label: mean_directional_error(
                            truth[class_label], prediction[class_label])
                        for class_label in truth.columns
                    }
                )
            else:
                metric['function'] = mean_directional_error
            metric['comparison'] = abs_lt
        elif metric['name'] == 'Visualization':
            metric['function'] = lambda truth, prediction: plot_average_value(
                truth,
                prediction,
                'Speaker',
                metric['threshold'],
            )

            metric['plot'] = True

        metric['display'] = lambda x: f'{x:.2f}'

    return databases, metrics


def test_metrics(
        metrics: typing.Sequence,
        task: str,
        truth: typing.Union[pd.DataFrame, pd.Series],
        prediction: typing.Union[pd.DataFrame, pd.Series],
        result_dir: str,
) -> typing.Dict:
    r"""Execute calculation of metrics.

    If the metric is a plot,
    it will store the plot file
    in ``result_dir``.

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

        # Run metric function to get scores
        result = metric['function'](truth, prediction)

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
